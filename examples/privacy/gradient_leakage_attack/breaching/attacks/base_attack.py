"""Base attack class."""
import copy
import mindspore as ms
from mindspore import ops
from .auxiliaries.common import optimizer_lookup


class _BaseAttacker:
    """Base attack class."""
    def __init__(self, model, loss_fn, cfg_attack):
        self.cfg = cfg_attack
        self.model_template = copy.copy(model)
        self.loss_fn = copy.copy(loss_fn)
        self.data_shape = None
        self.dm, self.ds = 0, 0
        self._rec_models = None

    def reconstruct(self, server_payload, shared_data, initial_data=None, custom=None):
        raise NotImplementedError()

    def prepare_attack(self, server_payload, shared_data):
        """prepare attack."""
        shared_data = shared_data.copy()
        server_payload = server_payload.copy()

        metadata = server_payload[0]["metadata"]
        self.data_shape = list(metadata.shape)
        if hasattr(metadata, "mean"):
            self.dm = ms.Tensor(list(metadata.mean))[None, :, None, None]
            self.ds = ms.Tensor(list(metadata.std))[None, :, None, None]
        else:
            self.dm, self.ds = ms.Tensor(0), ms.Tensor(1)

        rec_models = self._construct_models_from_payload_and_buffers(server_payload, shared_data)
        shared_data = self._cast_shared_data(shared_data)
        self._rec_models = rec_models

        if shared_data[0]["metadata"]["labels"] is None:
            labels = self._recover_label_information(shared_data)
        else:
            labels = copy.copy(shared_data[0]["metadata"]["labels"])
        labels = labels.astype(ms.int32)
        return rec_models, labels

    def _construct_models_from_payload_and_buffers(self, server_payload, shared_data):
        """construct models."""
        models = []
        for idx, payload in enumerate(server_payload):

            new_model = copy.copy(self.model_template)

            # Load parameters
            parameters = payload["parameters"]
            if shared_data[idx]["buffers"] is not None:
                buffers = shared_data[idx]["buffers"]
                new_model.set_train(False)
            elif payload["buffers"] is not None:
                buffers = payload["buffers"]
                new_model.set_train(False)
            else:
                new_model.set_train(True)
                buffers = []

            for param, server_state in zip(new_model.trainable_params(), parameters):
                param.set_data(server_state.value())
            for buffer, server_state in zip(new_model.untrainable_params(), buffers):
                buffer.set_data(server_state.value())
            models.append(new_model)
        return models

    def _cast_shared_data(self, shared_data):
        """cast data"""
        for data in shared_data:
            data["gradients"] = [g for g in data["gradients"]]
            if data["buffers"] is not None:
                data["buffers"] = [b for b in data["buffers"]]
        return shared_data

    def _initialize_data(self, data_shape):
        """initialize data"""
        init_type = self.cfg.init
        if init_type == "randn":
            candidate = ops.randn(data_shape)
        elif init_type == "randn-trunc":
            candidate = (ops.randn(data_shape) * 0.1).clamp(-0.1, 0.1)
        elif init_type == "rand":
            candidate = (ops.rand(data_shape) * 2) - 1.0
        elif init_type == "zeros":
            candidate = ops.zeros(data_shape)
        else:
            raise ValueError(f"Unknown initialization scheme {init_type} given.")

        candidate.requires_grad = True
        return candidate

    def _init_optimizer(self, candidate):
        """init optimizer"""
        optimizer, scheduler = optimizer_lookup(
            candidate,
            self.cfg.optim.optimizer,
            self.cfg.optim.step_size,
            scheduler=self.cfg.optim.step_size_decay,
            max_iterations=self.cfg.optim.max_iterations,
        )
        return optimizer, scheduler

    def _recover_label_information(self, user_data):
        """recover label information"""
        num_data_points = user_data[0]["metadata"]["num_data_points"]
        num_classes = user_data[0]["gradients"][-1].shape[0]

        if self.cfg.label_strategy is None:
            return None
        if self.cfg.label_strategy == "iDLG":
            label_list = []
            for shared_data in user_data:
                last_weight_min = ms.ops.argmin(ops.sum(shared_data["gradients"][-2], dim=-1), axis=-1)
                label_list += [last_weight_min.detach()]
            labels = ops.stack(label_list)
        elif self.cfg.label_strategy == "yin":
            total_min_vals = ms.Tensor(0)
            for shared_data in user_data:
                total_min_vals += shared_data["gradients"][-2].min(axis=-1)
            labels = total_min_vals.argsort()[:num_data_points]
            print(labels.shape)
        else:
            raise ValueError(f"Invalid label recovery strategy {self.cfg.label_strategy} given.")

        if len(labels) < num_data_points:
            labels = ms.ops.concat(
                [labels, ms.ops.randint(0, num_classes, (num_data_points-len(labels)))]
            )

        labels = (labels.sort()[0]).astype(ms.int32)
        print(f"Recovered labels {labels}.")
        return labels
