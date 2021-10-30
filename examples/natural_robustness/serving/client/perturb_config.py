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

"""
Configuration of natural robustness methods for server.
"""

PerturbConfig = [{"method": "Contrast", "params": {"alpha": 1.5, "beta": 0}},
                 {"method": "GaussianBlur", "params": {"ksize": 5}},
                 {"method": "SaltAndPepperNoise", "params": {"factor": 0.05}},
                 {"method": "Translate", "params": {"x_bias": 0.1, "y_bias": -0.2}},
                 {"method": "Scale", "params": {"factor_x": 0.7, "factor_y": 0.7}},
                 {"method": "Shear", "params": {"factor": 2, "direction": "horizontal"}},
                 {"method": "Rotate", "params": {"angle": 40}},
                 {"method": "MotionBlur", "params": {"degree": 5, "angle": 45}},
                 {"method": "GradientBlur", "params": {"point": [50, 100], "kernel_num": 3, "center": True}},
                 {"method": "GradientLuminance", "params": {"color_start": [255, 255, 255], "color_end": [0, 0, 0],
                                                            "start_point": [100, 150], "scope": 0.3,
                                                            "bright_rate": 0.3, "pattern": "light", "mode": "circle"}},
                 {"method": "GradientLuminance", "params": {"color_start": [255, 255, 255],
                                                            "color_end": [0, 0, 0], "start_point": [150, 200],
                                                            "scope": 0.3, "pattern": "light", "mode": "horizontal"}},
                 {"method": "GradientLuminance", "params": {"color_start": [255, 255, 255], "color_end": [0, 0, 0],
                                                            "start_point": [150, 200], "scope": 0.3,
                                                            "pattern": "light", "mode": "vertical"}},
                 {"method": "Perlin", "params": {"ratio": 0.5, "shade": 0.1}},
                 {"method": "Curve", "params": {"curves": 10, "depth": 10, "mode": "vertical"}},
                 {"method": "BackgroundWord", "params": {"shade": 0.1}},
                 {"method": "Perspective", "params": {"ori_pos": [[0, 0], [0, 800], [800, 0], [800, 800]],
                                                      "dst_pos": [[50, 0], [0, 800], [780, 0], [800, 800]]}},
                 {"method": "BackShadow", "params": {"back_type": 'leaf', "shade": 0.2}},
                 {"method": "BackShadow", "params": {"back_type": 'window', "shade": 0.2}},
                 {"method": "BackShadow", "params": {"back_type": 'person', "shade": 0.1}},
                 {"method": "BackShadow", "params": {"back_type": 'background', "shade": 0.1}},
                 ]
