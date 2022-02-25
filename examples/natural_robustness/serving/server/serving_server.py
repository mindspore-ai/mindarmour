# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
"""The server of example perturbation"""

import os
import sys
from mindspore_serving import server


def start():
    """Start server."""
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

    servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="perturbation",
                                                 device_ids=(0, 1), num_parallel_workers=4)
    server.start_servables(servable_configs=servable_config)

    server.start_grpc_server(address="0.0.0.0:5500", max_msg_mb_size=200)
    # server.start_restful_server(address="0.0.0.0:5500")


if __name__ == "__main__":
    start()
