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
"""underlying implementation of single-input functional encryption"""
import logging
import math

import gmpy2 as gp
from crypto.utils import _param_generator
from crypto.utils import _random
from crypto.utils import _random_generator
from crypto.utils import load_sec_param_config

logger = logging.getLogger(__name__)


class SIFEDynamicTPA:
    """Class for Single-Input Functional Encryption with dynamic third-party authority."""

    def __init__(self, eta, sec_param=256, sec_param_config=None):
        self.eta = eta
        if sec_param_config is not None:
            self.p, self.q, self.r, self.g, self.sec_param = load_sec_param_config(sec_param_config)
        else:
            self.p, self.q, self.r = _param_generator(sec_param)
            self.g = _random_generator(sec_param, self.p, self.r)
            self.sec_param = sec_param

    def setup(self):
        self.msk = [_random(self.p, self.sec_param) for i in range(self.eta)]
        pk = [gp.powmod(self.g, self.msk[i], self.p) for i in range(self.eta)]
        self.mpk = {'p': self.p, 'g': self.g, 'pk': pk}

    def generate_common_public_key(self):
        pk = dict()
        pk['g'] = gp.digits(self.mpk['g'])
        pk['p'] = gp.digits(self.mpk['p'])
        return pk

    def generate_public_key(self, vec_size):
        assert vec_size <= self.eta
        pk = dict()
        pk['bound'] = vec_size
        pk['g'] = gp.digits(self.mpk['g'])
        pk['p'] = gp.digits(self.mpk['p'])
        pk['pk'] = list()
        for i in range(vec_size):
            pk['pk'].append(gp.digits(self.mpk['pk'][i]))
        return pk

    def generate_private_key(self, vec):
        assert len(vec) <= self.eta

        sk = gp.mpz(0)
        for i in range(len(vec)):
            sk = gp.add(sk, gp.mul(self.msk[i], vec[i]))
        return {'bound': len(vec), 'sk': gp.digits(sk)}


class SIFEDynamicClient:
    """Class for Single-Input Functional Encryption with dynamic client functionality."""

    def __init__(self, sec_param=256, role='dec', dlog=None):
        if role not in ('dec', 'both'):
            if dlog is not None:
                self.dlog_table = dlog['dlog_table']
                self.func_bound = dlog['func_bound']
            else:
                self.sec_param = sec_param
                self.dlog_table = None
                self.func_bound = None
        elif role == 'enc':
            self.sec_param = sec_param

    def encrypt(self, pk, vec):
        """Encrypt the plaintext using the public key."""
        assert len(vec) == pk['bound']

        p = gp.mpz(pk['p'])
        g = gp.mpz(pk['g'])

        r = _random(p, self.sec_param)
        ct0 = gp.digits(gp.powmod(g, r, p))
        ct_list = []
        for i in range(len(vec)):
            ct_list.append(gp.digits(
                gp.mul(
                    gp.powmod(gp.mpz(pk['pk'][i]), r, p),
                    gp.powmod(g, gp.mpz(int(vec[i])), p)
                )
            ))
        return {'ct0': ct0, 'ct_list': ct_list}

    def decrypt(self, pk, sk, vec, ct, max_innerprod):
        """Decrypt the ciphertext using the secret key."""
        p = gp.mpz(pk['p'])
        g = gp.mpz(pk['g'])

        res = gp.mpz(1)
        for i in range(len(vec)):
            res = gp.mul(
                res,
                gp.powmod(gp.mpz(ct['ct_list'][i]), gp.mpz(vec[i]), p)
            )
        res = gp.t_mod(res, p)
        g_f = gp.divm(res, gp.powmod(gp.mpz(ct['ct0']), gp.mpz(sk['sk']), p), p)

        f = self._solve_dlog(p, g, g_f, max_innerprod)

        return f

    def _solve_dlog(self, p, g, h, dlog_max):
        """
        Attempts to solve for the discrete log x, where g^x = h mod p via
        hash table.
        """
        if self.dlog_table is not None and str(h) in self.dlog_table:
            return self.dlog_table[str(h)]
        logger.warning("did not find f in dlog table, may cost more time to compute")
        return self._solve_dlog_naive(p, g, h, dlog_max)

    def _solve_dlog_naive(self, p, g, h, dlog_max):
        """
        Attempts to solve for the discrete log x, where g^x = h, via
        trial and error. Assumes that x is at most dlog_max.
        """
        res = None
        for j in range(dlog_max):
            if gp.powmod(g, j, p) == gp.mpz(h):
                res = j
                break
        if res is None:
            h = gp.invert(h, p)
            for i in range(dlog_max):
                if gp.powmod(g, i, p) == gp.mpz(h):
                    res = -i
        return res

    def _solve_dlog_bsgs(self, g, h, p):
        """
        Attempts to solve for the discrete log x, where g^x = h mod p,
        via the Baby-Step Giant-Step algorithm.
        """
        m = math.ceil(math.sqrt(p - 1))  # phi(p) is p-1, if p is prime
        # store hashmap of g^{1,2,...,m}(mod p)
        hash_table = {pow(g, i, p): i for i in range(m)}
        # precompute via Fermat's Little Theorem
        c = pow(g, m * (p - 2), p)
        # search for an equivalence in the table. Giant Step.
        for j in range(m):
            y = (h * pow(c, j, p)) % p
            if y in hash_table:
                return j * m + hash_table[y]
        return None
