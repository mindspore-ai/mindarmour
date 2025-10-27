# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/09/05 16:45:50
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : Attack.py
# @Description  : 这里使用策略模式抽象出所有策略的共有方式attack()
from abc import ABC, abstractmethod
# class Attack(ABC):
#     def __init__(self, name = None):
#         self.name = name
        
#     def get_attack_name(self):
#         if self.name is not None:
#             return self.name 
    
#     @abstractmethod
#     def attack():
#         raise NotImplementedError

class Attack(ABC):
    @abstractmethod
    def create_poisoned_dataset(self,dataset):
        raise NotImplementedError


