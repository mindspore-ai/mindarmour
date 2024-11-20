# Inference Theft Module Guide

## Overview

The inference theft module implements model stealing attacks and defenses during the inference phase. It includes three main methods: KnockOff (attack), MAZE (attack), and PVMTA (defense).

## Attack Methods

### 1. KnockOff Attack

KnockOff is a model stealing attack that queries the target model with samples from public datasets to replicate its functionality.

**Key Features:**

- Uses public datasets for training data
- Applicable to both general and traffic scenarios
- Simple but effective stealing mechanism

**Usage Example:**

```python
knockoff = Knockoff(
    teacher_model=target_model,
    student_model=clone_model,
    seed=42,
    batch_size=128,
    learning_rate=0.01,
    epochs=100,
    optimizer_name="adam"
)

# Execute attack
knockoff(surrogate_loader=public_dataset,
         test_loader=test_dataset)

# Get the stolen model
stolen_model = knockoff.get_student_model()
```

### 2. MAZE Attack

MAZE is an advanced model stealing attack that uses generative models to compensate for lack of training data knowledge.

**Key Features:**

- Uses generative models for synthetic data
- Supports zero-order gradient estimation
- Includes generator and discriminator models
- Allows experience replay for better training

**Usage Example:**

```python
maze = MAZE(
    teacher_model=teacher_model,
    student_model=clone_model,
    generator_model=generator,
    discriminator_model=discriminator,
    batch_size=128,
    lr_student=0.01,
    lr_generator=0.001,
    lr_discriminator=0.001,
    optimizer_name="adam",
    budget=3e7,
    latent_dim=128
)

# Execute attack
maze(generator_train_loader=gen_dataset)
```

## Defense Method

### PVMTA (Prediction-phase Vulnerability Mitigation Through Temperature Adjustment)

PVMTA is a defense mechanism that dynamically adjusts softmax temperature to reduce information leakage.

**Key Features:**

- Dynamic temperature adjustment
- No additional training required
- Minimal impact on legitimate users
- Significant reduction in stealing efficiency

**Usage Example:**

```python
pvmta = PVMTA(
    teacher_model=target_model,
    student_model=clone_model,
    seed=42,
    batch_size=128,
    learning_rate=0.01,
    epochs=100,
    surrogate_data_loader=public_dataset,
    optimizer_name="adam"
)

# Execute defense
pvmta(test_loader=test_dataset)
```

## Implementation Details

### Base Class

The InferenceTheftBase class provides common functionality for all methods. Key implementations include:

Reference to base class methods:

```python

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
```

### KnockOff Implementation

Reference to KnockOff implementation:

```python
class Knockoff(InferenceTheftBase):
    """KnockOff attack implementation.

    This class implements the KnockOff attack which steals model functionality by querying
    the target model with samples from public datasets.

    Args:
        teacher_model: Target model to be stolen from
        student_model: Model that will learn to replicate teacher behavior
        seed: Random seed for reproducibility
        batch_size: Number of samples per batch
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        optimizer_name: Name of optimizer to use, either 'sgd' or 'adam'
    """
    def __init__(
        self,
        teacher_model: nn.Cell,
        student_model: nn.Cell,
        seed: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        optimizer_name: Literal['sgd', 'adam'] = 'sgd',
    ):
        super().__init__()

        self.teacher_model = teacher_model
        self.student_model = student_model

        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.optimizer_name = optimizer_name

    def __call__(self, surrogate_loader: Dataset, test_loader: Dataset):
        """Execute KnockOff attack training process.

        Args:
            surrogate_loader: DataLoader containing surrogate training data
            test_loader: DataLoader containing test data for evaluation
        """
        self.teacher_model.set_train(False)
        self.student_model.set_train(True)

        if self.optimizer_name == 'sgd':
            opt = nn.SGD(self.student_model.trainable_params(),
                        learning_rate=self.learning_rate,
                        momentum=0.9,
                        weight_decay=5e-4)
        elif self.optimizer_name == 'adam':
            opt = nn.Adam(self.student_model.trainable_params(),
                         learning_rate=self.learning_rate,
                         weight_decay=5e-4)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not implemented")

        print('== Training Student Model ==')

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_soft_epoch(
                self.student_model, opt, surrogate_loader)
            _, test_acc = self.test_epoch(self.student_model, test_loader)

            print(f'Epoch: {epoch+1} Loss: {train_loss:.4f} '
                  f'Train Acc: {train_acc:.2f}% Test Acc: {test_acc:.2f}%')

        print('== Student Model Training Done ==')

    def get_student_model(self):
        """Return the trained student model.

        Returns:
            nn.Cell: Trained student model
        """
        return self.student_model
```

### MAZE Implementation

Reference to MAZE core functionality:

```python
...
class MAZE(InferenceTheftBase):
    """MAZE attack implementation.

    This class implements the MAZE attack which steals model functionality by using
    generative models to compensate for lack of training data knowledge.
    """
    def __init__(
        self,
        teacher_model: nn.Cell,
        student_model: nn.Cell,
        generator_model: nn.Cell,
        discriminator_model: nn.Cell,
        batch_size: int,
        lr_student: float,
        lr_generator: float,
        lr_discriminator: float,
        optimizer_name: Literal['sgd', 'adam'] = 'sgd',
        budget: int = 3e7,
        iters_student: int = 5,
        iters_generator: int = 1,
        iters_exp: int = 10,
        latent_dim: int = 128,
        num_dirs: int = 10,
        alpha_gan: float = 0.0,
    ):
        """
        Initialize MAZE attack.

        Args:
            teacher_model: Target model to be stolen from
            student_model: Model that will learn to replicate teacher behavior
            generator_model: Generator model to create synthetic training data
            discriminator_model: Discriminator model for adversarial training
            batch_size: Number of samples per batch
            lr_student: Learning rate for student model
            lr_generator: Learning rate for generator model
            lr_discriminator: Learning rate for discriminator model
            optimizer_name: Name of optimizer to use, either 'sgd' or 'adam'
            budget: Total number of queries allowed
            iters_student: Number of student update iterations per round
            iters_generator: Number of generator update iterations per round
            iters_exp: Number of experience replay iterations
            latent_dim: Dimension of latent space for generator
            num_dirs: Number of random directions for gradient estimation
            alpha_gan: Weight for generator loss
        """

        super().__init__()

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model

        self.batch_size = batch_size
        self.lr_student = lr_student
        self.lr_generator = lr_generator
        self.lr_discriminator = lr_discriminator
        self.optimizer_name = optimizer_name

        self.budget = float(budget)
        self.iters_student = iters_student
        self.iters_generator = iters_generator
        self.iters_exp = iters_exp
        self.latent_dim = latent_dim
        self.num_dirs = num_dirs
        self.alpha_gan = alpha_gan

        self.budget_per_iter = self.batch_size * (
            (self.iters_student - 1) + (1 + self.num_dirs) * self.iters_generator
        )
```

### PVMTA Defense Implementation

Reference to PVMTA implementation:

```python
class PVMTA(Knockoff):
    """PVMTA defense class that inherits from Knockoff.

    This class implements the PVMTA defense mechanism by adjusting softmax temperature
    dynamically during prediction to mitigate model stealing attacks.
    """
    def __init__(
        self,
        teacher_model: nn.Cell,
        student_model: nn.Cell,
        seed: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        surrogate_data_loader: Dataset,
        optimizer_name: Literal['sgd', 'adam'] = 'sgd'
    ):
        """Initialize PVMTA defense.

        Args:
            teacher_model: Target model to be protected
            student_model: Clone model used by attacker
            seed: Random seed for reproducibility
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            surrogate_data_loader: Data loader for surrogate dataset
            optimizer_name: Name of optimizer to use, either 'sgd' or 'adam'
        """
        super().__init__(
            teacher_model,
            student_model,
            seed=seed,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            optimizer_name=optimizer_name
        )

        self.surrogate_data_loader = GeneratorDataset(
            source=lambda: self.defensive_fn(surrogate_data_loader, teacher_model),
            column_names=["data", "label"],
            column_types=[ms.float32, ms.float32]
        )

    def defensive_fn(self, surrogate_data_loader: Dataset, target_model: nn.Cell):
        """Defensive function that adjusts temperature dynamically.

        Args:
            surrogate_data_loader: Data loader for surrogate dataset
            target_model: Target model to protect

        Yields:
            Tuple of (data, perturbed_output)
        """
        temperature = Tensor(1.0)

        for _, (datum, _) in enumerate(surrogate_data_loader):
            for _ in range(128):
                if random.random() > 0.3:
                    temperature *= Tensor(0.9)

            target_outs = target_model(datum)
            target_outs = ops.softmax(target_outs / temperature, axis=1)
            target_outs = ms.numpy.rand(*target_outs.shape)

            for data, target_out in zip(datum, target_outs):
                yield (data, target_out)

    def __call__(self, test_loader: GeneratorDataset):
        """Run the defense.

        Args:
            test_loader: Data loader for test dataset
        """
        super().__call__(self.surrogate_data_loader, test_loader)
```

## Supported Datasets

1. General Scenarios:
  - CIFAR
  - SVHN
  - FashionMNIST
2. Traffic Scenarios:
  - GTSRB

## Best Practices

1. For KnockOff:
  - Use diverse public datasets
  - Adjust learning rate based on convergence
  - Monitor student model accuracy
2. For MAZE:
  - Tune generator and discriminator learning rates
  - Adjust latent dimension based on data complexity
  - Balance generator and student training iterations
3. For PVMTA:
  - Start with default temperature settings
  - Monitor legitimate user performance
  - Adjust confidence thresholds as needed
4. General:
  - Always validate defense effectiveness
  - Monitor query budget consumption
  - Balance security and utility trade-offs
