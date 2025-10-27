# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/09/05 21:20:43
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py
# @Description  : xxxx

# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/09/05 16:45:50
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : Attack.py
# @Description  : 这里使用策略模式抽象出所有策略的共有方式attack()

from abc import ABC, abstractmethod
class Defense(ABC):
    @abstractmethod
    def repair(self, dataset=None, schedule=None):
        raise NotImplementedError
    # # if the defense strategy is based data filter,then it must finish method
    # @abstractmethod
    # def filter(self, dataset=None, schedule=None):
    #     pass


