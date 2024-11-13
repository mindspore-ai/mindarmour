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
import os
import cv2
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.communication as comm


def create_dataset(dataset_type, dataset_path, batch_size, train):
    
    rank_id = comm.get_rank()
    rank_size = comm.get_group_size()

    # ============================================================================
    # CIFAR10
    # ============================================================================
    if dataset_type == 'CIFAR10':
        if train:
            dataset = ds.Cifar10Dataset(dataset_path, num_shards=rank_size, shard_id=rank_id, usage='train')
        else:
            dataset = ds.Cifar10Dataset(dataset_path, num_shards=rank_size, shard_id=rank_id, usage='test')

        image_transforms = [
            ds.vision.Rescale(1.0 / 255.0, 0),
            ds.vision.HWC2CHW()
        ]
        label_transform = ds.transforms.TypeCast(ms.int32)
        dataset = dataset.map(image_transforms, 'image')
        dataset = dataset.map(label_transform, 'label')

    # ============================================================================
    # SVHN
    # ============================================================================
    elif dataset_type == 'SVHN':
        if train:
            dataset = ds.SVHNDataset(dataset_path, num_shards=rank_size, shard_id=rank_id, usage='train')
        else:
            dataset = ds.SVHNDataset(dataset_path, num_shards=rank_size, shard_id=rank_id, usage='test')

        image_transforms = [
            ds.vision.Rescale(1.0 / 255.0, 0),
            # ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
            ds.vision.HWC2CHW()
        ]
        label_transform = ds.transforms.TypeCast(ms.int32)
        dataset = dataset.map(image_transforms, 'image')
        dataset = dataset.map(label_transform, 'label')

    # ============================================================================
    # MedMNIST
    # ============================================================================
    elif dataset_type == 'MedMNIST':
        dataset_name = "pathmnist"   # num_classes = 9
        data_file = os.path.join(dataset_path, f"{dataset_name}.npz")
        data = np.load(data_file)

        if train:
            train_images = data["train_images"]
            train_labels = data["train_labels"].reshape(-1)
            my_accessible = MyAccessible(train_images, train_labels)
            dataset = ds.GeneratorDataset(source=my_accessible, column_names=["image", "label"], 
                                          num_shards=rank_size, shard_id=rank_id)
        else:
            test_images = data["test_images"]
            test_labels = data["test_labels"].reshape(-1)
            my_accessible = MyAccessible(test_images, test_labels)
            dataset = ds.GeneratorDataset(source=my_accessible, column_names=["image", "label"], 
                                          num_shards=rank_size, shard_id=rank_id)

        image_transforms = [
            ds.vision.Rescale(1.0 / 255.0, 0),
            # ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
            ds.vision.HWC2CHW()
        ]
        label_transform = ds.transforms.TypeCast(ms.int32)
        dataset = dataset.map(image_transforms, 'image')
        dataset = dataset.map(label_transform, 'label')

    # ============================================================================
    # Flower102
    # ============================================================================
    elif dataset_type == 'Flower102':
        if train:
            dataset = ds.Flowers102Dataset(dataset_dir=dataset_path,
                                        task="Classification", 
                                        usage="train", 
                                        decode=False)
            
            image_transforms = [
                ds.vision.Decode(),
                ds.vision.Resize((224, 224)),
                ds.vision.Rescale(1.0 / 255.0, 0),
                ds.vision.HWC2CHW()
            ]
            label_transform = ds.transforms.TypeCast(ms.int32)
            dataset = dataset.map(image_transforms, 'image')
            dataset = dataset.map(label_transform, 'label')

        else:
            dataset = ds.Flowers102Dataset(dataset_dir=dataset_path,
                                        task="Classification", 
                                        usage="test", 
                                        decode=True)

    dataset = dataset.batch(batch_size)
    return dataset


class MyAccessible():
    def __init__(self, data, label):
        self._data = data
        self._label = label

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)
    

def generator(indices, image_folder, labels):
    
    for idx in indices:
        image_path = os.path.join(image_folder, "image_{}.jpg".format(str(idx).zfill(5)))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = labels[idx]

        print(type(image), type(label))
        
        yield image, label


if __name__ == '__main__':
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    comm.init()
    ms.set_seed(1)

    dataset_path = "./dataset/CIFAR10"
    dataset = create_dataset('CIFAR10', dataset_path, batch_size=32, train=True)
    print(dataset.output_shapes())
    print(dataset.get_dataset_size())
    print(dataset.num_classes())


