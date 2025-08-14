# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn
import mindspore.communication as comm

from model_load import resnet18
from data_load import create_dataset


# ============================================================================
# Define some important functions
# ============================================================================

# Define Flatten function
def flatten(tensors):
    if len(tensors) == 1:
        flat = tensors[0].view(-1)
    else:
        flat = ops.cat([t.view(-1) for t in tensors], axis=0)
    return flat


# Define Unflatten function
def unflatten(flat, tensors):
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


# Define Gaussian Noise Generating function
def generate_gaussian_noise(length, mean, std):
    gaussian_noise = ms.Tensor(np.random.normal(mean, std, length), ms.float32)
    return gaussian_noise


# Define Forward function
def forward_fn(data, label):
    output = model(data)
    loss = criterion(output, label)
    return loss, output


# ============================================================================
# Distributed and Communication Initialization
# ============================================================================
ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
comm.init()
ms.set_seed(1)
np.random.seed(1)
gpu = comm.get_rank()
num_workers = comm.get_group_size()


if __name__ == "__main__":
    """
    The users only need to modify the following region according to their training tasks.
    """
    # ==================================================================================================
    epochs = 10
    lr = 0.01
    sigma = 0.008
    train_batch_size = 64
    test_batch_size = 100

    # Model and Optimizer Initialization
    model = resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.SGD(model.trainable_params(), learning_rate=lr)

    # Estimate maximum gradient norm
    gradient_norm_bound_estimate = 5

    # Data Load
    dataset_path = "./dataset/CIFAR10"
    train_dataset = create_dataset('CIFAR10', dataset_path, batch_size=train_batch_size, train=True)
    test_dataset = create_dataset('CIFAR10', dataset_path, batch_size=test_batch_size, train=False)
    # ==================================================================================================
    """
    The users only need to modify the above region according to their training tasks.
    """

    # Define Gradient Computation and Communication
    grad_fn = ms.value_and_grad(forward_fn, None, model.trainable_params(), has_aux=True)
    grad_reducer = nn.DistributedGradReducer(optimizer.parameters)
    all_sum = ops.AllReduce(op="sum")

    # Training and Testing Pipeline
    for epoch in range(epochs):
        # Train pipeline
        model.set_train()
        for iteration, (inputs, targets) in enumerate(train_dataset):
            output = model(inputs)
            (loss, _), grads = grad_fn(inputs, targets)  # loss and gradients
            flatten_grads = flatten(grads)

            if epoch == 0 and iteration == 0:
                length = flatten_grads.numel()
                local_s = ms.Tensor(np.zeros(length), ms.float32)
                local_v = ms.Tensor(np.zeros(length), ms.float32)
                global_s = ms.Tensor(np.zeros(length), ms.float32)
                global_v = ms.Tensor(np.zeros(length), ms.float32)

            standard_vr = sigma * gradient_norm_bound_estimate
            noise = generate_gaussian_noise(flatten_grads.numel(), 0, standard_vr)
            flatten_grads += noise
            before_sparsify = flatten_grads.copy() - local_s.copy()
            num_zeros = int(length * 0.1)
            zero_indices_np = np.random.choice(length, num_zeros, replace=False)
            zero_indices_ms = ms.Tensor(zero_indices_np, ms.int32)
            updates = ms.Tensor(np.zeros(num_zeros, dtype=np.float32), ms.float32)
            local_v = ops.tensor_scatter_elements(before_sparsify, zero_indices_ms, updates, axis=0)
            local_s += 0.5 * local_v.copy()
            unflattened_local_v = unflatten(local_v, grads)
            all_reduced_local_v = grad_reducer(unflattened_local_v)
            global_v = global_s.copy() + flatten(all_reduced_local_v)
            global_s += 0.5 * flatten(all_reduced_local_v)
            unflattened_global_v = unflatten(global_v, grads)
            optimizer(unflattened_global_v)  # update

        print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, loss.item()))

        # Test pipeline
        model.set_train(False)
        total = 0
        correct = 0
        for step, (inputs, targets) in enumerate(test_dataset):
            outputs = model(inputs)
            _, predicted = ops.max(outputs, 1)
            total += len(targets)
            correct += (predicted == targets).sum().item()
        print('Epoch: {}, Accuracy: {:.4f}'.format(epoch, 100 * correct / total))


