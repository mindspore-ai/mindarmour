# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/21 10:25:23
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py
# @Description  : xxxx
from abc import ABC, abstractmethod
from copy import deepcopy
from .Attack import Attack
from ..Base import *
from .BadNets import BadNets
#这里根据策略设计模式，也可以这样设计：
# （1）令一个个策略直接继承Attack和badnets，然后依据具体的攻击策略实现Attack接口的方法
# （2）设计一个backdoor类作为策略的context，聚合所有的策略，起到承上启下的封装作用，屏蔽高层模块的对策略、算法的直接访问，
# 封装可能的存在的变化

class BackdoorAttack(object):
    """
    As the context class of the strategy mode, this class aggregates all specific attack strategies, acts as a link between 
    the preceding and the following, and shields high-level modules from direct access to strategies and algorithms.
    It could have several methods:
        1. return poisoning datasets 
        2. return backdoor Model

    According to actual needs, other methods can be defined.

    Args:
        attack_method(Attack):a specific attack strategy object.
        
    Attributes:
        poisoned_train_dataset(torch.utils.data.Dataset) : poisoning training dataset
        poisoned_test_dataset(torch.utils.data.Dataset) : poisoning test dataset
        backdoor_model(torch.nn.Module): The resulting backdoor model after training on the poisoning training dataset.
    """
    def __init__(self, attack_method):
        self.attack_method = attack_method
        self.clean_train_dataset, self.clean_test_dataset = self.attack_method.get_dataset()
        self.poisoned_train_dataset = self.poisoned_test_dataset = None
        self.backdoor_model = None

    def get_attack_strategy(self):
        return self.attack_method.get_attack_strategy()

    def get_backdoor_model(self):
        return self.attack_method.get_model()
    
    def create_poisoned_train_dataset(self, y_target = None, poisoned_rate = None):
        self.poisoned_train_dataset = self.attack_method.create_poisoned_dataset(deepcopy(self.clean_train_dataset), y_target=y_target, poisoned_rate=poisoned_rate, train=True) 
        return self.poisoned_train_dataset
 
    def create_poisoned_test_dataset(self, y_target = None, poisoned_rate = None):
        self.poisoned_test_dataset = self.attack_method.create_poisoned_dataset(deepcopy(self.clean_test_dataset), y_target=y_target, poisoned_rate=poisoned_rate, train=False)
        return self.poisoned_test_dataset
    
    # 这里预留一个接口，对于定义好的对象，可以针对不同的数据集产生不同的有毒数据集
    # def get_poisoned_dataset(self, dataset=None, schedule=None):
    #     return self.train_dataset
    
    #根据具体的攻击策略发动攻击，攻击训练允许自定义schedule。如果schedule=None，则默认使用定义attack_method对象时指定的调度
    def attack(self, train_dataset = None, test_dataset=None, schedule = None):
           self.attack_method.train(train_dataset=train_dataset, test_dataset=test_dataset, schedule=schedule)
        
    # 这里测试逻辑允许指定调度和测试数据集
    def test(self, model=None, test_dataset=None, schedule=None): 
        return self.attack_method.test(model, test_dataset, schedule)







 







 
