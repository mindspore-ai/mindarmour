# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
from pathlib import Path
from string import Template
import numpy as np
import ml_dtypes
from safetensors import safe_open
from safetensors.numpy import save_file

from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Model Protection'

_supported_config_type = [
    'obf_metadata_config',
    'weight_obf_config',
    'network_obf_config'
]

_supported_metadata_type = [
    'random',
    'rearrange'
]

_supported_model_format = [
    'safetensors'
]

def _check_dir_path(name, dir_path):
    """check directory path"""
    if not isinstance(dir_path, str):
        raise TypeError("{} must be string, but got {}.".format(name, type(dir_path)))
    if not os.path.exists(dir_path):
        raise ValueError("{} is not exist, please check the input {}.".format(dir_path, name))
    if not Path(dir_path).is_dir():
        raise TypeError("{} must be a directory path, but got {}.".format(name, dir_path))

def _check_ops_info(ops_info):
    """check ops info config"""
    for op in ops_info:
        op_name = op.get('name')
        if not isinstance(op_name, str):
            raise TypeError("op_name type should be str, but got {}.".format(type(op_name)))
        input_x_name = op.get('input_x')
        if not isinstance(input_x_name, str):
            raise TypeError("input_x_name type should be str, but got {}.".format(type(input_x_name)))
        input_y_name = op.get('input_y')
        if not isinstance(input_y_name, str):
            raise TypeError("input_y_name type should be str, but got {}.".format(type(input_y_name)))
        if not isinstance(op.get('transpose_a', False), bool):
            raise TypeError("transpose_a type should be bool, but got {}.".format(type(op.get('transpose_a'))))
        if not isinstance(op.get('transpose_b', False), bool):
            raise TypeError("transpose_b type should be bool, but got {}.".format(type(op.get('transpose_b'))))

def _check_new_input_info(insert_new_input):
    """check new input config"""
    if not isinstance(insert_new_input, list):
        raise TypeError("obf_config[][]['insert_new_input'] type should be list, but got {}."
                        .format(type(insert_new_input)))
    for new_input in insert_new_input:
        input_name = new_input.get('name')
        if not isinstance(input_name, str):
            raise TypeError("obf_config[][]['insert_new_input'][]['name'] type should be str, but got {}."
                            .format(type(input_name)))


def _check_obf_metadata_config(config):
    """check obf metadata config"""
    name = config.get('name')
    if not name or not isinstance(name, str):
        raise TypeError("obf_config[][]['name'] type should be str, but got {}.".format(type(name)))
    shape = config.get('shape')
    if not shape or not isinstance(shape, list):
        raise TypeError("obf_config[][]['shape'] type should be list, but got {}.".format(type(shape)))
    for item in shape:
        if not isinstance(item, int):
            raise TypeError("shape[] type should be int, but got {}.".format(type(item)))
    save_metadata = config.get('save_metadata', True)
    if not isinstance(save_metadata, bool):
        raise TypeError("obf_config[][]['save_metadata'] type should be bool, but got {}."
                        .format(type(save_metadata)))
    metadata_type = config.get('type')
    if metadata_type is not None:
        if not isinstance(metadata_type, str) or metadata_type not in _supported_metadata_type:
            raise TypeError("obf_config[][]['type'] should be str and must in {}, but got {}."
                            .format(str(_supported_metadata_type), type(metadata_type)))


def _check_weight_obf_config(config):
    """check weight obfuscation config"""
    target = config.get('target')
    if not target or not isinstance(target, str):
        raise TypeError("obf_config[][]['target'] type should be str, but got {}.".format(type(target)))
    weight_obf_ops = config.get('weight_obf_ops', [])
    if not isinstance(weight_obf_ops, list):
        raise TypeError("obf_config[][]['weight_obf_ops'] type should be list, but got {}."
                        .format(type(weight_obf_ops)))
    _check_ops_info(weight_obf_ops)


def _check_network_obf_config(config):
    """check network obfuscation config"""
    target = config.get('target')
    if not target or not isinstance(target, str):
        raise TypeError("obf_config[][]['target'] type should be str, but got {}.".format(type(target)))
    module = config.get('module')
    if not module or not isinstance(module, str):
        raise TypeError("obf_config[][]['module'] type should be str, but got {}.".format(type(module)))
    insert_new_input = config.get('insert_new_input', [])
    _check_new_input_info(insert_new_input)
    insert_ops = config.get('insert_ops', [])
    if not isinstance(insert_ops, list):
        raise TypeError("obf_config[][]['insert_ops'] type should be list, but got {}.".format(type(insert_ops)))
    _check_ops_info(insert_ops)


def _check_valid_obf_config(obf_config, config_type):
    """check obfuscation config"""
    if not isinstance(config_type, str) or config_type not in _supported_config_type:
        raise TypeError("config_type must be str, and in {}, but got {}."
                        .format(str(_supported_config_type), config_type))
    for config_type_item in obf_config.keys():
        if not isinstance(config_type_item, str) or config_type_item not in _supported_config_type:
            raise TypeError("config_type must be str, and in {}, but got {}."
                            .format(str(_supported_config_type), config_type_item))
    config_list = obf_config.get(config_type)
    if not isinstance(config_list, list):
        raise TypeError("obf_config[] type of should be list, but got {}.".format(type(config_list)))

    for config in config_list:
        if not isinstance(config, dict):
            raise TypeError("obf_config[][] type should be dict, but got {}.".format(type(config)))
        if config_type == 'obf_metadata_config':
            _check_obf_metadata_config(config)
        elif config_type == 'weight_obf_config':
            _check_weight_obf_config(config)
        elif config_type == 'network_obf_config':
            _check_network_obf_config(config)
        layers = config.get('layers')
        if layers is not None:
            if not isinstance(layers, list):
                raise TypeError("obf_config[][]['layers'] type should be list, but got {}.".format(type(layers)))
            for layer in layers:
                if not isinstance(layer, int):
                    raise TypeError("obf_config[][]['layers'][] type should be int, but got {}.".format(type(layer)))
    return True

def _check_valid_obf_config(obf_config):
    """check obfuscation config"""
    if isinstance(obf_config, dict):
        for config_type in obf_config.keys():
            if not isinstance(config_type, str) or config_type not in _supported_config_type:
                raise TypeError("config_type must be str, and in {}, but got {}."
                                .format(str(_supported_config_type), config_type))
            config_list = obf_config.get(config_type)
            if not isinstance(config_list, list):
                raise TypeError("obf_config[] type of should be list, but got {}.".format(type(config_list)))
            for config in config_list:
                if not isinstance(config, dict):
                    raise TypeError("obf_config[][] type should be dict, but got {}.".format(type(config)))
                if config_type == 'obf_metadata_config':
                    _check_obf_metadata_config(config)
                elif config_type == 'weight_obf_config':
                    _check_weight_obf_config(config)
                elif config_type == 'network_obf_config':
                    _check_network_obf_config(config)
                layers = config.get('layers')
                if layers is not None:
                    if not isinstance(layers, list):
                        raise TypeError("obf_config[][]['layers'] type should be list, but got {}.".format(type(layers)))
                    for layer in layers:
                        if not isinstance(layer, int):
                            raise TypeError("obf_config[][]['layers'][] type should be int, but got {}.".format(type(layer)))
    return obf_config

class ModelObfuscator:
    """
        Obfuscate the plaintext model weight files according to the obfuscation config.

        Args:
            obf_config (dict, optional): The configuration of model obfuscation polices. Default: ``None``.
            obfuscate_scale (Union[float, int], optional): Obfuscate scale of weights.
                The generated random obf_ratios will be in
                range of (1 / obfuscate_scale, obfuscate_scale). Default: ``100``.


        Examples:
            >>> from mindarmour import ModelObfuscator
            >>> obf_config_str = ""
            >>> src_path = ""
            >>> saved_path = ""
            >>> obf_config = yaml.safe_load(obf_config_str)
            >>> obf = ModelObfuscator(obf_config, obfuscate_scale=100)
            >>> obf.obfuscate_weight_files(src_path, saved_path=saved_path)
        """

    def __init__(self, obf_config, obfuscate_scale):
        self.obf_config = _check_valid_obf_config(obf_config)
        self.obf_metadata_config = self.obf_config.get('obf_metadata_config', [])
        self.obfuscate_scale = obfuscate_scale
        self.obf_metadata, self.saved_metadata = self._gen_obf_metadata(self.obf_metadata_config)
        self.saved_metadata_mapping = {}

    def _gen_obfuscate_tensor(self, tensor_shape, tensor_type='rearrange'):
        obf_tensor = None
        if tensor_type == 'rearrange':
            if len(tensor_shape) == 1:
                obf_tensor = np.random.permutation(tensor_shape[0])
            if len(tensor_shape) == 2:
                tensor = np.identity(tensor_shape[0])
                p = np.random.permutation(tensor_shape[1])
                obf_tensor = tensor[:, p]
        if tensor_type == 'random':
            obf_tensor = np.random.randint(1, self.obfuscate_scale, size=tensor_shape).astype(np.float16)
        return obf_tensor
    
    def _get_real_name(self, src_name, tag, substitute):
        dst_names = []
        for sub in substitute:
            strTemplate = Template(src_name)
            dst_name = strTemplate.safe_substitute({tag: str(sub)})
            dst_names.append(dst_name)
        return dst_names

    def _gen_obf_metadata(self, obf_metadata_config):
        obf_metadata = {}
        saved_metadata = {}
        for config in obf_metadata_config:
            name = config.get('name')
            if name is None:
                return
            save_metadata = config.get('save_metadata', False)
            metadata_op_name = config.get('metadata_op')
            layers = config.get('layers')  
            
            if not layers:
                if not obf_metadata.get(name):
                    obf_tensor = self._gen_obfuscate_tensor(config.get('shape'), config.get('type'))
                    obf_metadata[name] = obf_tensor
                    if save_metadata:
                        saved_metadata[name] = obf_tensor
            else:
                for layer in layers:
                    strTemplate = Template(name)
                    obf_name = strTemplate.safe_substitute({"layer": str(layer)})
                    obf_tensor = self._gen_obfuscate_tensor(config.get('shape'), config.get('type'))
                    obf_metadata[obf_name] = obf_tensor
                    if save_metadata:
                        saved_metadata[name] = obf_tensor
        return obf_metadata, saved_metadata
    
    def set_metadata(self, new_metadata):
        if not isinstance(new_metadata, dict):
            raise ValueError("xxx")
        self.obf_metadata = {k: new_metadata[k] if k in new_metadata else self.obf_metadata[k] for k in self.obf_metadata}
        self.saved_metadata = {k: new_metadata[k] if k in new_metadata else self.saved_metadata[k] for k in self.saved_metadata}
    
    def set_save_metadata_mapping(self, new_mapping):
        self.saved_metadata_mapping.update(new_mapping)

    def _obfuscate_params(self, params, obf_metadata, obf_config, not_obfuscated_params):
        """Obfuscate params"""
        def _get_op_input_name(obf_op, name_key='input_x', layer=0):
            op_name = obf_op.get('name')
            input_name = obf_op.get(name_key)
            if input_name is None:
                LOGGER.error(TAG, "can not find input: {} for op: {}.".format(name_key, op_name))
                return None
            strTemplate = Template(input_name)
            input_name = strTemplate.safe_substitute({"layer": str(layer)})
            return input_name

        def _get_op_input(input_name, obf_param):
            op_input = obf_metadata.get(input_name, None) if input_name.startswith('obf_metadata') else obf_param
            return op_input

        def _obfuscate_param(param, obf_metadata, obf_ops, layer=0):
            param_dtype = param.dtype
            obf_param = param
            for obf_op in obf_ops:
                op_name = obf_op.get('name')
                if not isinstance(op_name, str):
                    raise TypeError('{} should be str type, but got {}'.format(op_name, type(op_name)))
                if op_name == 'mul':
                    input_x = obf_param
                    input_y_name = _get_op_input_name(obf_op, 'input_y', layer)
                    input_y = obf_metadata.get(input_y_name)
                    if input_x is None or input_y is None:
                        LOGGER.error(TAG, "input_x or input_y is None")
                        return None
                    input_y = input_y.astype(param_dtype)
                    obf_param = np.multiply(input_x, input_y)
                elif op_name == 'permuate':
                    index_name = _get_op_input_name(obf_op, 'input_y', layer)
                    p = obf_metadata.get(index_name, None)
                    axis = obf_op.get('axis')
                    if p is None or obf_param is None or axis is None:
                        LOGGER.error(TAG, "input_x or param or axis is None")
                        return None
                    if axis == 0:
                        obf_param = obf_param[p]
                    elif axis == 1:
                        obf_param = obf_param[:, p]
                    else:
                        raise ValueError('axis should be 0 or 1, but got {}'.format(axis))
                elif op_name == 'matmul':
                    input_x_name = _get_op_input_name(obf_op, 'input_x', layer)
                    input_y_name = _get_op_input_name(obf_op, 'input_y', layer)
                    input_x = _get_op_input(input_x_name, obf_param)
                    input_y = _get_op_input(input_y_name, obf_param)
                    if input_x is None or input_y is None:
                        LOGGER.error(TAG, "the input_x or input_y of op: {} is None.".format(op_name))
                        return None
                    input_x = np.transpose(input_x, (1, 0)) if obf_op.get('transpose_a', False) else input_x
                    input_y = np.transpose(input_y, (1, 0)) if obf_op.get('transpose_b', False) else input_y
                    obf_param = np.matmul(input_x.astype(param_dtype), input_y.astype(param_dtype))
                else:
                    LOGGER.error(TAG, "unsupported op, op must be matmul or permuate or mul, but got {}."
                                .format(op_name))
                    return None
            return obf_param


        weight_obf_config = obf_config.get('weight_obf_config', [])
        for item, param in params.items():
            item_split = item.split('.')
            param_path = '/'.join(item_split[:len(item_split)])
            for obf_target in weight_obf_config:
                if not isinstance(obf_target, dict):
                    raise TypeError('{} should be dict type, but got {}'.format(obf_target, type(obf_target)))
                target = obf_target.get('target', None)
                layers = obf_target.get('layers', [])
                obf_ops = obf_target.get('weight_obf_ops', None)
                if not target or not obf_ops:
                    raise KeyError("target or obf_ops is None.")
                if not layers:
                    if target == param_path:
                        obf_param = _obfuscate_param(param, obf_metadata, obf_ops)
                        if obf_param is None:
                            LOGGER.error(TAG, "obfuscate weight {} failed.".format(item))
                            return False
                        params[item] = obf_param
                        LOGGER.info(TAG, "obfuscate weight: {} success.".format(item))
                        not_obfuscated_params.remove(item)
                for layer in layers:
                    strTemplate = Template(target)
                    target_path = strTemplate.safe_substitute({"layer": str(layer)})
                    if target_path == param_path:
                        obf_param = _obfuscate_param(param, obf_metadata, obf_ops, layer)
                        if obf_param is None:
                            LOGGER.error(TAG, "obfuscate weight {} failed.".format(item))
                            return False
                        params[item] = obf_param
                        LOGGER.info(TAG, "obfuscate weight: {} success.".format(item))
                        not_obfuscated_params.remove(item)
        return True

    def _obfuscate_safetensor_files(self, src_path, saved_path='./'):
        params_not_obfuscated = []
        file_names = os.listdir(src_path)
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {}
        }
        add_obf_metadata = True
        for file_name in file_names:
            if not file_name.endswith('.safetensors'):
                continue
            not_obfuscated = self._obfuscate_single_file(os.path.join(os.path.realpath(src_path), file_name),
                                                         self.obf_metadata, self.obf_config, saved_path, index, add_obf_metadata)
            params_not_obfuscated += not_obfuscated
            add_obf_metadata = False
        
        LOGGER.info(TAG, "params not obfuscated: {}.".format(params_not_obfuscated))
        param_json_path = os.path.join(saved_path, "model.safetensors.index.json")
        with open(param_json_path, "w") as f:
            json.dump(index, f, indent=2)

    def obfuscate_weight_files(self, src_path, saved_path='./', file_format="safetensors"):
        """
        Obfuscate the plaintext model weight files according to the obfuscation config.

        Args:
            src_path (str): The directory path of original model weight files.
            saved_path (str, optional): The directory path for saving obfuscated model files. Default: ``'./'``.

        Returns:
            dict[str], obf_metadata, which is the necessary data that needs to be load when running obfuscated network.

        Raises:
            TypeError: If `src_path` is not string or `saved_path` is not string.
            ValueError: If `src_path` is not exist or `saved_path` is not exist.
        """

        _check_dir_path('src_path', src_path)
        _check_dir_path('saved_path', saved_path)
        
        if file_format == "safetensors":
            self._obfuscate_safetensor_files(src_path, saved_path)
        else:
            LOGGER.error(TAG, "got unsupported model format: {}. Currently supported model formats include {}."
                        .format(file_format, _supported_model_format))

    def _obfuscate_single_file(self, src_file, obf_metadata, obf_config, saved_path, index, add_obf_metadata):
        """Obfuscate single model file"""
        
        params = {}
        not_obfuscated_params = []
        file_name = src_file.split('/')[-1]
        try:
            with safe_open(src_file, framework="np") as f:
                for param_name in f.keys():
                    #print(hf_param_name)
                    params[param_name] = f.get_tensor(param_name)
                    index["weight_map"][param_name] = file_name
                    not_obfuscated_params.append(param_name)
        except (ValueError, TypeError, OSError):
            LOGGER.error(TAG, "Load params failed for file {}.".format(src_file))
            return not_obfuscated_params
        
        LOGGER.info(TAG, "Load params success for file {}.".format(src_file))
        self._obfuscate_params(params, obf_metadata, obf_config, not_obfuscated_params)

        # save the obfuscated hg to saved_path
        if add_obf_metadata:
            for param_name in self.saved_metadata_mapping:
                metadata_name = self.saved_metadata_mapping[param_name]
                params[param_name] = self.saved_metadata[metadata_name]
                index["weight_map"][param_name] = file_name
        obf_file_name = os.path.join(saved_path, file_name)
        LOGGER.info(TAG, "Saving obfuscated params to file: {}.".format(obf_file_name))
        save_file(params, obf_file_name)
        index["metadata"]["total_size"] += os.path.getsize(obf_file_name)
        return not_obfuscated_params
