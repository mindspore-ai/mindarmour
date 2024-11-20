"""
KnockOff: Anti-theft during inference phase, improves theft attack efficiency by sampling from
public datasets. Supports PyTorch, MindSpore, PaddlePaddle, TensorFlow. Applicable to general
scenarios (CIFAR, SVHN, FashionMNIST) and traffic scenarios (GTSRB).
"""

from typing import Literal
from mindspore import nn
from mindspore.dataset import Dataset

from .base import InferenceTheftBase

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
