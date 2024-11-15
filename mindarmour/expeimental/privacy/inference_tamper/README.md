# Inference Tampering Module Guide

## Overview

The inference tampering module implements backdoor attacks and defenses for neural networks during the inference phase. It includes two attack methods (BadNets and Blended) and one defense mechanism (STRIP).

## Attack Methods

### 1. BadNets Attack

BadNets is a straightforward backdoor attack that directly injects visible triggers into input images.

**Key Features:**

- Directly pastes visible triggers in a specified region
- Supports both clean and dirty label attacks
- Applicable to CIFAR10 and GTSRB datasets

**Usage Example:**

```python
badnets = BadNets(
    target_model=model,
    dataset=train_dataset,
    poison_ratio=0.1,  # 10% of dataset poisoned
    image_size=32,
    trigger_size=4,
    target_label=0,
    label_mode="clean"  # or "dirty"
)

# Train the model with poisoned data
badnets(
    loss_fn=nn.CrossEntropyLoss(),
    optimizer_name="Adam",
    learning_rate=0.01,
    epochs=100
)

# Get the poisoned model
poisoned_model = badnets.get_poisoned_model()
```

### 2. Blended Attack

Blended attack is a more subtle approach that mixes trigger patterns with original images using specified blending weights.

**Key Features:**

- Blends trigger patterns with original images
- Adjustable blending weight
- Less visible than BadNets
- Supports both clean and dirty label attacks

**Usage Example:**

```python
blended = Blended(
    target_model=model,
    dataset=train_dataset,
    poison_ratio=0.1,
    image_size=32,
    trigger_size=4,
    target_label=0,
    label_mode="clean",
    weight=0.2  # blending weight
)

# Train with poisoned data
blended(
    loss_fn=nn.CrossEntropyLoss(),
    optimizer_name="Adam",
    learning_rate=0.01,
    epochs=100
)
```

## Defense Method

### STRIP Defense

STRIP is a defense mechanism that detects poisoned images by analyzing their prediction entropy under perturbations.

**Key Features:**

- Blends test images with clean images
- Uses prediction entropy as detection metric
- Lower entropy indicates higher probability of being adversarial
- No model modification required

**Usage Example:**

```python
strip = STRIP(
    target_model=model,
    benign_dataset=clean_dataset,
    transparency=0.5
)

# Test an image
entropy = strip(test_image, k=10)  # k is number of blending operations

# Lower entropy indicates potential backdoor
is_poisoned = entropy < threshold
```

## Implementation Details

### Base Attack Class

The `TriggerAttackBase` class provides common functionality for both attacks:

Reference to base implementation:

```python
...
class TriggerAttackBase:
    """Base class for trigger attacks like BadNets and Blended.
    BadNets and Blended share most functionality except trigger pattern.
    Dataset should be in shape (H,W,C) with pixel values in [0,1].

    Args:
        target_model (nn.Cell): Target model to attack
        dataset (ImageFolderDataset): Dataset to poison
        poison_ratio (float): Ratio of samples to poison, between 0 and 1
        image_size (int): Size of input images
        trigger_size (int): Size of trigger pattern
        target_label (int): Target label for poisoned samples
        label_mode (str): "clean" to only poison samples of target class,
                         "dirty" to poison samples of all classes
    """
    def __init__(
            self,
            target_model: nn.Cell,
            dataset: ImageFolderDataset,
            poison_ratio: float,
            image_size: int,
            trigger_size: int,
            target_label: int,
            label_mode: Literal["clean", "dirty"] = "clean"
    ):
        self.target_model = target_model
        self.dataset = dataset
        self.poison_ratio = poison_ratio
        self.image_size = image_size
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.label_mode = label_mode

        poisoned_indices_len = int(self.poison_ratio * len(self.dataset))
        to_poison_indices = self.get_dataset_indices()

        if poisoned_indices_len > len(to_poison_indices):
            raise ValueError(
                f"Poisoned indices length {poisoned_indices_len} exceeds dataset length"
            )

        self.poisoned_indices = set(
            random.sample(to_poison_indices, k=poisoned_indices_len)
        )

    def get_dataset_indices(self):
        """Get indices of dataset samples to poison.

        Returns:
            list: List of indices to poison based on label_mode
        """
        if self.label_mode == "dirty":
            return list(range(len(self.dataset)))
        if self.label_mode == "clean":
            return [i for i, (_, label) in enumerate(self.dataset)
                   if label == self.target_label]
        raise ValueError(f"Invalid label_mode: {self.label_mode}")

    def poison_dataset(self, dataset):
        """Generate poisoned samples from dataset.

        Args:
            dataset: Dataset to poison

        Yields:
            tuple: (poisoned_image, poisoned_label) pairs
        """
        for idx, (image, label) in enumerate(dataset):
            if idx in self.poisoned_indices:
                image = self.add_trigger(image)
                label = self.target_label
```

### STRIP Defense Implementation

The STRIP defense uses entropy-based detection:

Reference to STRIP implementation:

```python
class STRIP:
    """STRIP defense algorithm class

    Detects adversarial samples by calculating prediction entropy of blended images.
    Lower entropy indicates higher probability of being an adversarial sample.

    Args:
        target_model (nn.Cell): Target model to defend
        benign_dataset (Dataset): Clean dataset for blending
        transparency (float): Blending transparency, default 0.5
    """
    def __init__(
        self,
        target_model: nn.Cell,
        benign_dataset: Dataset,
        transparency: float = 0.5,
    ):
        self.target_model = target_model
        self.benign_dataset = benign_dataset
        self.transparency = transparency
        self.target_model.set_train(False)

    @classmethod
    def blending(cls, background: Tensor, overlay: Tensor, transparency: float):
        """Blend two images

        Args:
            background: Background image
            overlay: Overlay image
            transparency: Transparency ratio

        Returns:
            Blended image
        """
        return (1 - transparency) * background + transparency * overlay

    def __call__(self, testing_image: Tensor, k: int = 10):
        """Execute STRIP detection

        Args:
            testing_image: Image to test
            k: Number of images to blend, default 10

        Returns:
            Average entropy value
        """
        batch_blended_images = []
        for index in random.sample(range(len(self.benign_dataset)), k):
            benign_image = self.benign_dataset[index][0]
            blended_image = self.blending(testing_image, benign_image, self.transparency)
            batch_blended_images.append(blended_image)

        # Convert to Tensor, shape: (k, channels, image_size, image_size)
        batch_blended_images = Tensor(batch_blended_images)

        # Get predictions
        logits = self.target_model(batch_blended_images)
        softmax = nn.Softmax(axis=1)
        probabilities = softmax(logits).numpy()
        entropy_sum = -np.nansum([p * np.log2(p) for p in probabilities])
        entropy_avg = entropy_sum / k

        return entropy_avg
```

## Supported Datasets

**General Scenarios:**

- CIFAR10

**Traffic Scenarios:**

- GTSRB (German Traffic Sign Recognition Benchmark)

## Best Practices

- Choose appropriate poison ratio (typically 0.1-0.2)
- For Blended attacks, use smaller blending weights (0.1-0.3) for better stealth
- For STRIP defense, adjust the number of blending operations (k) based on computational budget
- Use clean label mode when possible for more stealthy attacks
- Always validate the attack success rate on a clean test set
