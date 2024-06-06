"""optimization based attack."""
import math
import time
import os
import mindspore as ms
from mindspore import ops
from PIL import Image
from .base_attack import _BaseAttacker
from .auxiliaries.regularizers import regularizer_lookup
from .auxiliaries.objectives import objective_lookup

class OptimizationBasedAttacker(_BaseAttacker):
    """optimization based attack."""
    def __init__(self, model, loss_fn, cfg_attack):
        super().__init__(model, loss_fn, cfg_attack)
        self.txt_path = ''
        self.gt_img = None
        self.save_flag = 0
        objective_fn = objective_lookup.get(self.cfg.objective.type)
        if objective_fn is None:
            raise ValueError(f"Unknown objective type {self.cfg.objective.type} given.")
        self.objective = objective_fn(**self.cfg.objective)
        self.regularizers = []
        try:
            for key in self.cfg.regularization.keys():
                if self.cfg.regularization[key].scale > 0:
                    self.regularizers += [regularizer_lookup[key](**self.cfg.regularization[key])]
        except AttributeError:
            pass

    def reconstruct(self, server_payload, shared_data, initial_data=None, custom=None):
        print(f'initialized data: {initial_data}')
        rec_models, labels = self.prepare_attack(server_payload, shared_data)
        candidate_solutions = self._run_trial(rec_models, shared_data, labels, custom)
        reconstructed_data = dict(data=candidate_solutions, labels=labels)

        return reconstructed_data

    def _run_trial(self, rec_model, shared_data, labels, custom=None):
        """run attack."""
        open(self.txt_path, 'w', encoding='utf-8')
        self.gt_img = Image.open(self.cfg.save_dir + 'A_0.png')
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model)
        self.objective.initialize(self.cfg.impl, rec_model[0])
        candidate_all, minimal_list = [], []
        optimizer, scheduler = [], []
        for seed in range(1):
            candidate_tmp = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
            candidate_tmp = ms.Parameter(candidate_tmp)
            candidate_all.append(candidate_tmp)
            minimal_list.append(ms.Tensor(float("inf")))
            best_candidate = ops.randn_like(candidate_tmp)
            opt_tmp, sched_tmp = self._init_optimizer(candidate_all[seed])
            opt_tmp.update_parameters_name('data')
            optimizer.append(opt_tmp)
            scheduler.append(sched_tmp)
        current_wallclock = time.time()
        self.save_flag = 0

        for iteration in range(self.cfg.optim.max_iterations):
            for seed in range(1):
                candidate = candidate_all[seed]
                optim = optimizer[seed]
                grad_fn = ms.value_and_grad(self._forward_total_loss, None, optim.parameters)
                objective_value = self._get_obj_and_step(candidate, shared_data[0]["gradients"],
                                                         labels, iteration, grad_fn, optim, scheduler[seed])
                if self.cfg.optim.boxed:
                    max_p = (1 - self.dm) / self.ds
                    min_p = -self.dm / self.ds
                    candidate.set_data(ops.clamp(candidate, min_p, max_p))
                if "peroid_Add10" in self.cfg.objective.keys():
                    if objective_value < minimal_list[seed] or (iteration+1) % self.cfg.objective.peroid_Add10 == 0:
                        minimal_list[seed] = objective_value.item()
                        best_candidate = ms.Tensor(candidate)
                elif objective_value < minimal_list[seed]:
                    minimal_list[seed] = objective_value.item()
                    best_candidate = ms.Tensor(candidate)

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    obj_value = math.modf(float(objective_value))[0] + math.modf(float(objective_value))[1] % 10
                    print(f"{self.save_flag}|| It: {iteration + 1} | Rec. loss: {obj_value:2.4f} | "
                          f"T: {timestamp - current_wallclock:4.2f}s\n")
                    with open(self.txt_path, 'a', encoding='utf-8') as f:
                        f.write(
                            f"{self.save_flag}|| It: {iteration + 1} | Rec. loss: {obj_value:2.4f} | "
                            f"T: {timestamp - current_wallclock:4.2f}s\n"
                        )
                    current_wallclock = timestamp
                    if custom is not None:
                        if "save_dir" not in self.cfg.keys():
                            raise AttributeError('saving path is not given!!!!!!!!')
                        if not os.path.exists(self.cfg.save_dir):
                            os.mkdir(self.cfg.save_dir)
                        save_path = self.cfg.save_dir + f'recon_{iteration + 1}.png'
                        custom.save_recover(best_candidate, save_pth=save_path)
                        self.save_flag += 1
                if not ms.ops.isfinite(objective_value):
                    print(f"Recovery loss is non-finite in iteration {iteration}, seed {seed}. "
                          f"Cancelling reconstruction!")
                    break
        return best_candidate

    def _get_obj_and_step(self, candidate, shared_grads, labels, iteration, grad_fn, optim, sched):
        """one-step update."""
        total_objective, grad = grad_fn(candidate, shared_grads, labels, iteration)
        lr = sched(ms.Tensor(iteration))
        optim.learning_rate.set_data(lr)
        grad = grad[0]
        if self.cfg.optim.langevin_noise > 0:
            step_size = lr
            noise_map = ops.randn_like(candidate.value())
            grad += self.cfg.optim.langevin_noise * step_size * noise_map
        if self.cfg.optim.grad_clip is not None:
            grad_norm = candidate.value().norm()
            if grad_norm > self.cfg.optim.grad_clip:
                grad *= self.cfg.optim.grad_clip / (grad_norm + 1e-8)
        if self.cfg.optim.signed is not None:
            if self.cfg.optim.signed == "soft":
                scaling_factor = 1 - iteration / self.cfg.optim.max_iterations
                grad = ops.tanh(grad*scaling_factor)/scaling_factor
            elif self.cfg.optim.signed == "hard":
                grad = ops.sign(grad)
            else:
                pass
        grad = tuple([grad])
        optim(grad)
        return total_objective

    def _forward_total_loss(self, candidate, shared_grads, labels, iteration):
        objective_tmp = self.objective.get_matching_loss(shared_grads, candidate, labels)
        total_objective = 0
        total_objective += objective_tmp
        for regularizer in self.regularizers:
            total_objective += regularizer.construct_(candidate, iteration)
        return total_objective
