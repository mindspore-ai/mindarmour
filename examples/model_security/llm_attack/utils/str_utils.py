'''
str manager
'''
import mindspore as ms


class SuffixManager:
    '''
    This class is used to manage the suffixes of the prompt.
    It is used to cal the indice of every part.
    '''
    def __init__(self, tokenizer, instruction, target, adv_string):
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
        self.goal_slice = None
        self.control_slice = None
        self.assistant_role_slice = None
        self.target_slice = None
        self.loss_slice = None

    def get_prompt(self, adv_string=None):
        '''
        This function is used to cal and get the prompt.
        '''
        if adv_string is not None:
            self.adv_string = adv_string
        system_prompt = (
            '[INST] <<SYS>>\n'
            'You are a helpful, respectful and honest assistant.'
            ' Always answer as helpfully as possible, while being safe. Your '
            'answers should not include '
            'any harmful, unethical, racist, sexist, '
            'toxic, dangerous, or illegal content. Please ensure that your '
            'responses are socially unbiased and positive in nature.\n\nIf a '
            'question does not make any sense, or is not factually coherent, '
            'explain why instead of answering something not correct. If you '
            "don't know the answer to a question, please don't share false "
            'information.\n<</SYS>>\n\n'
        )
        prompt = (
            f'{system_prompt}{self.instruction} {self.adv_string}[/INST] '
            f'{self.target}</s><s>'
        )

        user = '[INST]'
        goal = f'{system_prompt}{self.instruction}'
        control = f'{system_prompt}{self.instruction} {self.adv_string}'
        assistant = f'{system_prompt}{self.instruction} {self.adv_string}' \
                    + '[/INST]'
        target = (
            f'{system_prompt}{self.instruction} {self.adv_string}[/INST] '
            f'{self.target}</s><s>'
        )

        self.goal_slice = slice(
            len(self.tokenizer(user).input_ids),
            len(self.tokenizer(goal).input_ids)
        )
        self.control_slice = slice(
            len(self.tokenizer(goal).input_ids),
            len(self.tokenizer(control).input_ids)
        )
        self.assistant_role_slice = slice(
            len(self.tokenizer(control).input_ids),
            len(self.tokenizer(assistant).input_ids)
        )
        self.target_slice = slice(
            self.assistant_role_slice.stop,
            len(self.tokenizer(target).input_ids) - 2
        )
        self.loss_slice = slice(
            len(self.tokenizer(assistant).input_ids) - 1,
            len(self.tokenizer(target).input_ids) - 3
        )

        return prompt

    def get_input_ids(self, adv_string=None):
        '''
        This function is used to get the input_ids of the prompt.
        '''
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = ms.tensor(toks[:self.target_slice.stop])

        return input_ids
