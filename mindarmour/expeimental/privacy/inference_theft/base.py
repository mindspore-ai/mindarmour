"""
Base class for inference theft attacks including KnockOff, MAZE and PVMTA.

KnockOff: Anti-theft during inference phase, improves theft attack efficiency by sampling from
public datasets. Supports PyTorch, MindSpore, PaddlePaddle, TensorFlow. Applicable to general
scenarios (CIFAR, SVHN, FashionMNIST) and traffic scenarios (GTSRB).

MAZE: Anti-theft during inference phase, uses generative models to compensate for lack of
training set knowledge. Supports PyTorch. Applicable to general scenarios (CIFAR, SVHN,
FashionMNIST) and traffic scenarios (GTSRB).

PVMTA: Anti-theft during inference phase, dynamically adjusts SoftMax temperature using
confidence to reduce information leakage. Supports PyTorch, MindSpore, PaddlePaddle,
TensorFlow. Applicable to general scenarios (CIFAR, SVHN, FashionMNIST) and traffic
scenarios (GTSRB).
"""

from typing import Literal
import numpy as np
import mindspore as ms
from mindspore import nn, Parameter, ops, Tensor
from mindspore.dataset import Dataset

class InferenceTheftBase:
    """Base class providing common methods for inference theft attacks."""
    def __init__(self):
        pass

    @classmethod
    def train_soft_epoch(cls, student_model: nn.Cell, opt: nn.Optimizer, train_loader: Dataset):
        """Train one epoch using soft labels.

        Args:
            opt: Optimizer for training
            train_loader: DataLoader containing training data

        Returns:
            tuple: (train_loss, train_accuracy)
        """
        criterion = nn.KLDivLoss(reduction='batchmean')

        def forward_fn(data, label):
            logits = student_model(data)
            preds_log = ops.log_softmax(logits, axis=1)
            loss = criterion(preds_log, label)
            return loss, logits

        grad_fn = ms.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)

        def train_step(data, label):
            (loss, output), grads = grad_fn(data, label)
            opt(grads)
            return loss, output

        correct = 0
        train_loss = 0
        total = 0

        for _, (data, target) in enumerate(train_loader.create_tuple_iterator()):
            total += len(data)
            loss, pred = train_step(data, target)
            train_loss += loss.asnumpy()
            correct += (pred.argmax(1) == target.argmax(1)).asnumpy().sum()

        train_loss /= len(train_loader)
        train_acc = correct * 100.0 / total
        return train_loss, train_acc

    @classmethod
    def test_epoch(cls, model: nn.Cell, test_loader: Dataset):
        """Run test epoch.

        Args:
            model: Model to test
            test_loader: Test data loader

        Returns:
            tuple: (test_loss, test_accuracy)
        """
        num_batches = test_loader.get_dataset_size()
        model.set_train(False)
        criterion = nn.NLLLoss(reduction='sum')

        test_loss = 0
        correct = 0
        total = 0

        for data, target in test_loader.create_tuple_iterator():
            pred = model(data)
            total += len(data)
            loss = criterion(pred, target).asnumpy()
            test_loss += loss
            correct += (pred.argmax(1) == target).asnumpy().sum()

        test_loss /= num_batches
        test_acc = correct * 100.0 / total

        model.set_train()
        return test_loss, test_acc

    @classmethod
    def kl_div_logits(cls, student_logits: Tensor, teacher_logits: Tensor,
                      reduction: Literal['batchmean', 'none'] = 'batchmean'):
        """Calculate KL divergence between student and teacher logits.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            reduction: Reduction method for KL divergence

        Returns:
            Tensor: KL divergence loss
        """
        divergence = -ops.kl_div(
            ops.log_softmax(student_logits, axis=1),
            ops.softmax(teacher_logits, axis=1),
            reduction=reduction)
        return divergence.sum(axis=1)

    @classmethod
    def generator_loss_noreduce(cls, student_logits: Tensor, teacher_logits: Tensor):
        """Calculate generator loss without reduction.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model

        Returns:
            Tensor: Generator loss
        """
        divergence = -ops.kl_div(
            ops.log_softmax(student_logits, axis=1),
            ops.softmax(teacher_logits, axis=1),
            reduction="none")
        return divergence.sum(axis=1)

    @classmethod
    def sur_stats(cls, student_logits: Tensor, teacher_logits: Tensor):
        """Calculate surrogate statistics between student and teacher predictions.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model

        Returns:
            tuple: (MSE loss, max difference mean, max prediction mean)
        """
        pred_student = nn.Softmax(axis=-1)(student_logits)
        pred_teacher = nn.Softmax(axis=-1)(teacher_logits)

        mse = nn.MSELoss()
        mse_loss = mse(pred_student, pred_teacher)

        max_diff = ops.reduce_max(ops.abs(pred_student - pred_teacher), axis=-1)
        max_pred = ops.reduce_max(pred_teacher, axis=-1)
        return mse_loss, max_diff.mean(), max_pred.mean()

    @classmethod
    def zoge_backward(cls, input_x: Tensor, input_x_pre: Tensor,
                      student_model: nn.Cell, teacher_model: nn.Cell,
                      num_dirs: int = 10, step_size: float = 0.001, batch_size: int = 128):
        """Zero-order gradient estimation backward pass.

        Args:
            input_x: Input tensor
            input_x_pre: Pre-processed input tensor
            student_model: Student model
            teacher_model: Teacher model
            num_dirs: Number of random directions
            step_size: Step size
            batch_size: Batch size

        Returns:
            tuple: (generator loss, cosine similarity, magnitude ratio)
        """
        for param in student_model.parameters_dict().values():
            if isinstance(param, Parameter):
                param.requires_grad = False

        grad_est = ops.zeros_like(input_x_pre)
        dim = np.array(input_x.shape[1:]).prod()

        student_outs = student_model(input_x)
        teacher_outs = teacher_model(input_x)

        if teacher_outs == 'Detected by PRADA':
            return False, False, False

        loss_generator = cls.generator_loss_noreduce(student_outs, teacher_outs)
        counter = 0

        for _ in range(num_dirs):
            rand_u = ops.randn(input_x_pre.shape)
            rand_u_flat = rand_u.view((batch_size, -1))
            rand_u_norm = rand_u / ops.LpNorm(p=2, axis=1)(rand_u_flat).view((-1, 1, 1, 1))

            x_mod_pre = input_x_pre + (step_size * rand_u_norm)
            x_mod = ops.tanh(x_mod_pre)

            student_outs = student_model(x_mod)
            teacher_outs = teacher_model(x_mod)
            if teacher_outs == 'Detected by PRADA':
                break

            loss_generator_mod = cls.generator_loss_noreduce(student_outs, teacher_outs)
            grad_est += (dim / num_dirs) * (loss_generator_mod - loss_generator) / (
                step_size * rand_u_norm
            )
            counter += 1

        grad_est /= counter
        grad_est /= batch_size

        x_det_pre = input_x_pre
        x_det = ops.tanh(x_det_pre)

        teacher_outs = teacher_model(x_det)
        if teacher_outs == 'Detected by PRADA':
            return False, False, False

        loss_generator_det = -cls.kl_div_logits(student_outs, teacher_outs)

        def gradient_model(input_tensor):
            """Calculate gradient for model."""
            x_det = ops.tanh(input_tensor)
            student_outs = student_model(x_det)
            teacher_outs = teacher_model(x_det)
            return -cls.kl_div_logits(student_outs, teacher_outs)

        grad = ms.grad(gradient_model)(x_det_pre)

        grad_true_flat = grad.view((batch_size, -1))
        grad_est_flat = grad_est.view((batch_size, -1))

        cosine_sim = ops.ReduceSum(axis=1)(grad_true_flat * grad_est_flat) / (
            ops.sqrt(ops.ReduceSum(axis=1)(ops.square(grad_true_flat))) *
            ops.sqrt(ops.ReduceSum(axis=1)(ops.square(grad_est_flat))))

        mag_ratio = ops.norm(grad_est_flat, dim=1) / ops.norm(grad_true_flat, dim=1)

        ops.composite.GradOperation(get_by_list=True)(grad_est, input_x_pre)

        for param in student_model.parameters_dict().values():
            if isinstance(param, Parameter):
                param.requires_grad = True

        return loss_generator_det.mean(), cosine_sim.mean(), mag_ratio.mean()
