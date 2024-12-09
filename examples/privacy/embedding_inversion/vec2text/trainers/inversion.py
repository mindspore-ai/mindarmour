'''
inversion trainer
'''
from datetime import datetime
from typing import Dict

import mindspore as ms

from trainers.base import BaseTrainer

class InversionTrainer(BaseTrainer):

    '''InversionTrainer'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model
        self.embedder = self.model.embedder
        self.counter = 0
        self.last_time_logger = datetime.now()
        self.each_time_logger = datetime.now()

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> ms.Tensor:
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def training_step(self, model: ms.nn.Cell, inputs: Dict[str, ms.Tensor]) -> ms.Tensor:
        """
        Performs a training step. we override to compute data-specific metrics.
        """
        # TODO: Log training metrics from below... (How to do with huggingface?)
        self.counter += 1
        print(self.counter, "  ", datetime.now() - self.each_time_logger)
        self.each_time_logger = datetime.now()

        if self.counter % 100 == 0:
            print("this 100 step consume:")
            print(datetime.now()-self.last_time_logger)
            self.last_time_logger = datetime.now()
        self._compute_data_metrics(inputs=inputs)
        # self.log({ f"train/{k}": v for k,v in metrics.items() })
        return super().training_step(model, inputs)


    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """Edit keys posthumously on model load."""
        # Rename keys for backward compatibility w/ model trained before
        # we added extra dropout to the model
        if {
                "embedding_transform.2.weight",
                "embedding_transform.2.bias",
        } <= state_dict.keys():
            print(
                "Renaming keys",
                {"embedding_transform.2.weight", "embedding_transform.2.bias"},
                "for backward compatibility.",
            )
            state_dict["embedding_transform.3.weight"] = state_dict.pop(
                "embedding_transform.2.weight"
            )
            state_dict["embedding_transform.3.bias"] = state_dict.pop(
                "embedding_transform.2.bias"
            )
        return state_dict

    #def _prepare_input(self, x):
        #return None
