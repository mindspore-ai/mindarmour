"""objective functions."""
import mindspore as ms
from mindspore import ops


class GradientLoss:
    """gradient matching loss."""
    def __init__(self):
        self.model = None
        self.cfg_impl = None

    def initialize(self, cfg_impl, model):
        self.model = model
        self.cfg_impl = cfg_impl

    def get_matching_loss(self, gradient_data, candidate, labels):
        gradient, _ = self._grad_fn_single_step(candidate, labels)
        objective = self.gradient_based_loss(gradient, gradient_data)
        return objective

    def gradient_based_loss(self, gradient_rec, gradient_data):
        raise NotImplementedError()

    def _grad_fn_single_step(self, candidate, labels):
        def _single_forward(candidate, labels):
            predict = self.model(candidate)
            task_loss = self._loss_fn(predict, labels)
            return task_loss
        grad_fn1 = ops.value_and_grad(_single_forward, None, weights=self.model.trainable_params())
        task_loss, gradient = grad_fn1(candidate, labels)
        return gradient, task_loss

    def _loss_fn(self, logits, labels):
        logs = ops.log(ops.softmax(logits))
        loss = self._nll_loss(logs, labels)
        return loss

    def _nll_loss(self, logs, labels):
        loss = [-logs[i, int(j)] for i, j in enumerate(labels)]
        loss = ops.stack(loss).mean()
        return loss


class Euclidean(GradientLoss):
    """euclidean distance."""
    def __init__(self, scale=1.0, start=0, min_start=0, broken_tail=0, peroid_Add10=1000, add10=10, **kwargs):
        super().__init__()
        self.tmp = kwargs
        self.scale = scale
        self.start = start
        self.min_start = min_start
        self.broken_tail = broken_tail
        self.peroid_add = peroid_Add10
        self.add10 = add10
        self.iter = 0

    def gradient_based_loss(self, gradient_rec, gradient_data):
        len_layer = len(gradient_data)
        final = len_layer
        if (self.iter + 1) % self.peroid_add == 0:
            self.start = self.start - self.add10 if (self.start - self.add10) > self.min_start else self.min_start
        self.iter += 1
        objective = 0
        for count, (rec, data) in enumerate(zip(gradient_rec, gradient_data)):
            if self.start <= count + 1 <= final:
                objective += (rec - data).pow(2).sum()
        return 0.5 * objective * self.scale

    @staticmethod
    def _euclidean(gradient_rec, gradient_data):
        """euclidean similarity."""
        objective = 0
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).pow(2).sum()
        return 0.5 * objective


class CosineSimilarity(GradientLoss):
    """cosine distance."""
    def __init__(self, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__()
        self.tmp = kwargs
        self.scale = scale
        self.task_regularization = task_regularization

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self._cosine_sim(gradient_rec, gradient_data) * self.scale

    @staticmethod
    def _cosine_sim(gradient_rec, gradient_data):
        """consine similarity."""
        scalar_product = ms.Tensor(0.0)
        rec_norm = ms.Tensor(0.0)
        data_norm = ms.Tensor(0.0)

        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.pow(2).sum()
            data_norm += data.pow(2).sum()

        objective = 1.0 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        return objective


class DynaLayerRandPickedCosineSimilarity(GradientLoss):
    """dynamic distance."""
    def __init__(self, scale=1.0, start=0, min_start=0, broken_tail=0, peroid_Add10=1000, add10=10, **kwargs):
        super().__init__()
        self.tmp = kwargs
        self.scale = scale
        self.start = start
        self.min_start = min_start
        self.broken_tail = broken_tail
        self.peroid_add = peroid_Add10
        self.add10 = add10
        self.iter = 0

    def gradient_based_loss(self, gradient_rec, gradient_data):
        len_layer = len(gradient_data)
        final = len_layer - self.broken_tail
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        if (self.iter+1) % self.peroid_add == 0:
            self.start = self.start-self.add10 if (self.start-self.add10) > self.min_start else self.min_start
        self.iter += 1
        for count, (rec, data) in enumerate(zip(gradient_rec, gradient_data)):
            if self.start <= count+1 <= final:
                mask = ms.ops.rand_like(data) > 0.0
                weight = 1.0
                scalar_product += (rec * data * mask).sum()*weight
                rec_norm += (rec * mask).pow(2).sum()*weight
                data_norm += (data * mask).pow(2).sum()*weight

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale



objective_lookup = {
    "euclidean": Euclidean,
    "cosine-similarity": CosineSimilarity,
    "dyna-layer-rand-cosine-similarity": DynaLayerRandPickedCosineSimilarity,
}
