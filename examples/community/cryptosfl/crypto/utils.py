# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""functions for the implementation in single-input functional encryption"""
import json
import logging
import random

import gmpy2 as gp

logger = logging.getLogger(__name__)


def _random(maximum, bits):
    rand_function = random.SystemRandom()
    r = gp.mpz(rand_function.getrandbits(bits))
    while r >= maximum:
        r = gp.mpz(rand_function.getrandbits(bits))
    return r


def _random_generator(bits, p, r):
    while True:
        h = _random(p, bits)
        g = gp.powmod(h, r, p)
        if not g == 1:
            break
    return g


def _random_prime(bits):
    rand_function = random.SystemRandom()
    r = gp.mpz(rand_function.getrandbits(bits))
    r = gp.bit_set(r, bits - 1)
    return gp.next_prime(r)


def _param_generator(bits, r=2):
    while True:
        p = _random_prime(bits)
        q = (p - 1) // 2
        if gp.is_prime(p) and gp.is_prime(q):
            break
    return p, q, r


def generate_config_files(sec_param, sec_param_config, dlog_table_config, func_bound):
    """
    Generate configuration files for secure parameters and discrete log tables.

    Args:
        sec_param (int): Security parameter.
        sec_param_config (str): Path to save the security parameter configuration.
        dlog_table_config (str): Path to save the discrete log table configuration.
        func_bound (int): Function bound for discrete log table.
    """
    p, q, r = _param_generator(sec_param)
    g = _random_generator(sec_param, p, r)
    group_info = {
        'p': gp.digits(p),
        'q': gp.digits(q),
        'r': gp.digits(r)
    }
    sec_param_dict = {'g': gp.digits(g), 'sec_param': sec_param, 'group': group_info}

    with open(sec_param_config, 'w') as outfile:
        json.dump(sec_param_dict, outfile)

    dlog_table = dict()
    bound = func_bound + 1
    for i in range(bound):
        dlog_table[gp.digits(gp.powmod(g, i, p))] = i
    for i in range(-1, -bound, -1):
        dlog_table[gp.digits(gp.powmod(g, i, p))] = i

    dlog_table_dict = {
        'g': gp.digits(g),
        'func_bound': func_bound,
        'dlog_table': dlog_table
    }

    with open(dlog_table_config, 'w') as outfile:
        json.dump(dlog_table_dict, outfile)


def load_sec_param_config(sec_param_config_file):
    """
    Load security parameter configuration from a file.

    Args:
        sec_param_config_file (str): Path to the security parameter configuration file.

    Returns:
        tuple: Contains p, q, r, g, and sec_param loaded from the configuration file.
    """
    with open(sec_param_config_file, 'r') as infile:
        sec_param_dict = json.load(infile)

        p = gp.mpz(sec_param_dict['group']['p'])
        q = gp.mpz(sec_param_dict['group']['q'])
        r = gp.mpz(sec_param_dict['group']['r'])
        g = gp.mpz(sec_param_dict['g'])
        sec_param = sec_param_dict['sec_param']

    return p, q, r, g, sec_param


def load_dlog_table_config(dlog_table_config_file):
    """
    Load discrete log table configuration from a file.

    Args:
        dlog_table_config_file (str): Path to the discrete log table configuration file.

    Returns:
        dict: Contains dlog_table, func_bound, and g loaded from the configuration file.
    """
    with open(dlog_table_config_file, 'r') as infile:
        store_dict = json.load(infile)

        dlog_table = store_dict['dlog_table']
        func_bound = store_dict['func_bound']
        g = gp.mpz(store_dict['g'])

    return {
        'dlog_table': dlog_table,
        'func_bound': func_bound,
        'g': g
    }
