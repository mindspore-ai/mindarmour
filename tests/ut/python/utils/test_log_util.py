# Copyright 2019 Huawei Technologies Co., Ltd
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
LogUtil test.
"""
import logging
import pytest

from mindarmour.utils.logger import LogUtil


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_logger():
    logger = LogUtil.get_instance()
    tag = 'Test'
    msg = 'i am %s, work in %s'
    args = ('tom', 'china')
    logger.debug(tag, msg, *args)
    logger.info(tag, msg, *args)
    logger.error(tag, msg, *args)
    logger.warn(tag, msg, *args)

    logger.set_level(logging.DEBUG)

    logger.debug(tag, msg, *args)
    logger.info(tag, msg, *args)
    logger.error(tag, msg, *args)
    logger.warn(tag, msg, *args)

    msg = 'i am tom, work in china'
    logger.info(tag, msg)

    logger.info(tag, 'accuracy is %f.', 0.995)
    logger.debug(tag, 'accuracy is %s.', 0.995)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_value_error():
    with pytest.raises(SyntaxError) as e:
        assert LogUtil()
    assert str(e.value) == 'can not instance, please use get_instance.'
    logger = LogUtil.get_instance()
    handler = 'logging.Handler(level=1)'
    with pytest.raises(ValueError):
        logger.add_handler(handler)
