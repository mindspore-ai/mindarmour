"""
PVMTA (Prediction-phase Vulnerability Mitigation Through Temperature Adjustment)
is a defense algorithm against model stealing attacks. It dynamically adjusts
the temperature of softmax to reduce information leakage during prediction phase,
which greatly reduces the efficiency of model stealing attacks.
"""

import random
from typing import Literal

from mindspore import nn, Tensor, ops
from mindspore.dataset import Dataset, GeneratorDataset
import mindspore as ms

from .knockoff import Knockoff

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
