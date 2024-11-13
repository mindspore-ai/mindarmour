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
"""generate the config of a discrete logarithm table"""
from crypto.utils import generate_config_files


def test_generate_config_files():
    sec_param_config_file = './sec_param.json'
    dlog_table_config_file = './dlog_b8.json'
    func_value_bound = 100000000
    sec_param = 256
    generate_config_files(sec_param, sec_param_config_file, dlog_table_config_file, func_value_bound)


if __name__ == "__main__":
    test_generate_config_files()
