from pathlib import Path
from typing import Union, Callable, Optional, Tuple, List, Any

from PIL import Image
import numpy as np
from mindspore.dataset import ImageFolderDataset, GeneratorDataset

from mindspore.dataset.vision import Resize, ToTensor

def default_loader(path: Union[str, Path]) -> Image.Image:
    """
    Load image from path and convert to RGB (3 channels), return in RGB mode (H, W, C).
    """
    img = Image.open(path).convert("RGB")
    
    return img

class Compose:
    """MindSpore-style Compose function, similar to torchvision.transforms.Compose"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n  {0}'.format(t)
        format_string += '\n)'
        return format_string


class DatasetFolder:
    """
    Mindspore does not support sample-level control dataset, so we need to implement our own DatasetFolder like torch does.
    """
    def __init__(
        self,
        root: Union[str, Path],
        loader: Callable = default_loader,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.class_to_idx = self._find_classes()
        # store samples(list): The path and class_index value for each image in the dataset
        self.samples = self._gather_samples()
        # store targets(list): The class_index value for each image in the dataset
        self.targets = [s[1] for s in self.samples]

    def _find_classes(self):
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def _gather_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        for cls_name, cls_idx in self.class_to_idx.items():
            for img_path in (self.root / cls_name).glob("*"):
                if img_path.suffix.lower() in self.extensions:
                    samples.append((img_path, cls_idx))
        return samples

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.uint32]:
        path, label = self.samples[index]
        
        image = self.loader(path)

        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image, dtype=np.uint8)

        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = np.int32(label)

        return image, label

    def __len__(self) -> int:
        return len(self.samples)

    def to_generator_dataset(self, batch_size=64, shuffle=True, num_parallel_workers=4):
        return GeneratorDataset(
            source=self,
            column_names=["image", "label"],
            shuffle=shuffle,
            num_parallel_workers=num_parallel_workers
        ).batch(batch_size=batch_size)


if __name__ == "__main__":
    # if it support four kind of datasets
    # 1. medmnist, /home/fz/.local/share/medmnist/train
    # 2. cifar100, /home/fz/.local/share/cifar100/train
    # 3. GTRSB, /home/fz/.local/share/GTRSB/Train
    # 4. flower102, /home/fz/.local/share/flower102/train
    import mindspore as ms
    
    ms.set_context(device_target="GPU", device_id=1)
    
    dataset_name = "flower102"
    root = f"/home/fz/.local/share/{dataset_name}/train" if dataset_name != "GTRSB" else f"/home/fz/.local/share/GTRSB/Train"
    
    # # Resize to 32 * 32, and totensor
    image_transform = Compose([
        Resize((32, 32)),
        ToTensor()
    ])
    
    # target_transform = Compose([
    #     ToTensor()
    # ])
    
    dataset = DatasetFolder(root=root, transform=image_transform)
    print(dataset[0][0], dataset[0][1])
    # for img, label in dataset:
    #     print(img.shape, label)
    
    
    
    # try dataloader
    for img, label in dataset.to_generator_dataset():
        print(img, label)
