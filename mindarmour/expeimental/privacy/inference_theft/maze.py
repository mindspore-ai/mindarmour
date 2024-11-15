"""
MAZE: Anti-theft during inference phase, uses generative models to compensate for lack of
training set knowledge. Supports PyTorch. Applicable to general scenarios (CIFAR, SVHN,
FashionMNIST) and traffic scenarios (GTSRB).
"""

from typing import Literal
import mindspore as ms
from mindspore import nn, Parameter, ops
from .base import InferenceTheftBase


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
        self.iters = int(self.budget / self.budget_per_iter)

    def __call__(self, generator_train_loader: ms.dataset.GeneratorDataset):
        """Execute MAZE attack training process.

        Args:
            generator_train_loader: DataLoader containing training data for generator
        """
        self.teacher_model.set_train(False)
        self.student_model.set_train()
        self.generator_model.set_train()
        self.discriminator_model.set_train()

        if self.optimizer_name == 'sgd':
            opt_student = nn.SGD(
                self.student_model.trainable_params(),
                lr=self.lr_student,
                momentum=0.9,
                weight_decay=5e-4
            )
            opt_generator = nn.SGD(
                self.generator_model.trainable_params(),
                lr=self.lr_generator,
                momentum=0.9,
                weight_decay=5e-4
            )

        elif self.optimizer_name == 'adam':
            opt_student = nn.Adam(
                self.student_model.trainable_params(),
                lr=self.lr_student,
                weight_decay=5e-4
            )
            opt_generator = nn.Adam(
                self.generator_model.trainable_params(),
                lr=self.lr_generator,
                weight_decay=5e-4
            )

        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not implemented")

        loss_generator = loss_generator_gan = loss_generator_dis = 0

        for param in self.teacher_model.parameters_dict().values():
            if isinstance(param, Parameter):
                param.requires_grad = False

        attacker_flag = False

        for _ in range(self.iters):
            # Update Generator
            for _ in range(self.iters_generator):
                if attacker_flag:
                    break

                latent = ops.randn((self.batch_size, self.latent_dim))
                images, images_pre = self.generator_model(latent)

                tmp1, _, _ = self.zoge_backward(
                    images,
                    images_pre,
                    self.student_model,
                    self.teacher_model
                )

                if tmp1 is False:
                    attacker_flag = True
                    teacher_outs = 'Detected by PRADA'
                    break

                loss_generator_dis = tmp1

                if loss_generator_dis is not False:
                    loss_generator = loss_generator_dis + (self.alpha_gan * loss_generator_gan)
                    opt_generator(loss_generator)

            # Update Student
            for iter_student in range(self.iters_student):
                if attacker_flag:
                    break

                if iter_student != 0:
                    latent = ops.randn((self.batch_size, self.latent_dim))
                    images, _ = self.generator_model(latent)

                teacher_outs = self.teacher_model(images)
                student_outs = self.student_model(images)

                if teacher_outs == 'Detected by PRADA':
                    break

                def forward_fn2(teacher_out, student_out):
                    loss_student = self.kl_div_logits(teacher_out, student_out)
                    return loss_student

                grad_fn = ms.value_and_grad(
                    forward_fn2,
                    None,
                    opt_student.parameters,
                    has_aux=False
                )

                loss_student_exp, grads = grad_fn(teacher_outs, student_outs)
                opt_student(grads)

            if not attacker_flag and teacher_outs != 'Detected by PRADA':

                gen_train_loader_iter = iter(generator_train_loader)

                loss_student_exp = 0.0

                def forward_fn4(student_model, images_prev, teacher_prev):
                    loss_student_exp = self.kl_div_logits(
                        student_model(images_prev),
                        teacher_prev
                    )
                    return loss_student_exp

                grad_fn = ms.value_and_grad(
                    forward_fn4,
                    None,
                    opt_student.parameters,
                    has_aux=False
                )

                def train_step4(student_model, images_prev, teacher_prev):
                    loss_student, grads = grad_fn(student_model, images_prev, teacher_prev)
                    opt_student(grads)
                    return loss_student

                for _ in range(self.iters_exp):
                    images_prev, _ = next(gen_train_loader_iter)

                    if images_prev.size(0) < self.batch_size:
                        break

                    loss_student_exp += train_step4(
                        self.student_model,
                        images_prev,
                        teacher_outs
                    )
                if self.iters_exp:
                    loss_student_exp /= self.iters_exp

            self.student_model.set_train()

    def get_student_model(self):
        """Return the trained student model.

        Returns:
            nn.Cell: Trained student model
        """
        return self.student_model
