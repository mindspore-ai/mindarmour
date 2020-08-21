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
""" Util for log module. """
import logging

_LOGGER = logging.getLogger('MA')


def _find_caller():
    """
    Bind findCaller() method, which is used to find the stack frame of the
    caller so that we can note the source file name, line number and
    function name.
    """
    return _LOGGER.findCaller()


class LogUtil:
    """
    Logging module.

    Raises:
        SyntaxError: If create this class.
    """
    _instance = None
    _logger = None
    _extra_fmt = ' [%s] [%s] '

    def __init__(self):
        raise SyntaxError('can not instance, please use get_instance.')

    @staticmethod
    def get_instance():
        """
        Get instance of class `LogUtil`.

        Returns:
            Object, instance of class `LogUtil`.
        """
        if LogUtil._instance is None:
            LogUtil._instance = object.__new__(LogUtil)
            LogUtil._logger = _LOGGER
            LogUtil._init_logger()
        return LogUtil._instance

    @staticmethod
    def _init_logger():
        """
        Initialize logger.
        """
        LogUtil._logger.setLevel(logging.WARNING)

        log_fmt = '[%(levelname)s] %(name)s(%(process)d:%(thread)d,' \
                  '%(processName)s):%(asctime)s%(message)s'
        log_fmt = logging.Formatter(log_fmt)

        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_fmt)

        # add the handlers to the logger
        LogUtil._logger.handlers = []
        LogUtil._logger.addHandler(console_handler)

        LogUtil._logger.propagate = False

    def set_level(self, level):
        """
        Set the logging level of this logger, level must be an integer or a
        string. Supported levels are 'NOTSET'(integer: 0), 'ERROR'(integer: 1-40),
        'WARNING'('WARN', integer: 1-30), 'INFO'(integer: 1-20) and 'DEBUG'(integer: 1-10).
        For example, if logger.set_level('WARNING') or logger.set_level(21), then
        logger.warn() and logger.error() in scripts would be printed while running,
        while logger.info() or logger.debug() would not be printed.

        Args:
            level (Union[int, str]): Level of logger.
        """
        self._logger.setLevel(level)

    def add_handler(self, handler):
        """
        Add other handler supported by logging module.

        Args:
            handler (logging.Handler): Other handler supported by logging module.

        Raises:
            ValueError: If handler is not an instance of logging.Handler.
        """
        if isinstance(handler, logging.Handler):
            self._logger.addHandler(handler)
        else:
            raise ValueError('handler must be an instance of logging.Handler,'
                             ' but got {}'.format(type(handler)))

    def debug(self, tag, msg, *args):
        """
        Log '[tag] msg % args' with severity 'DEBUG'.

        Args:
            tag (str): Logger tag.
            msg (str): Logger message.
            args (Any): Auxiliary value.
        """
        caller_info = _find_caller()
        file_info = ':'.join([caller_info[0], str(caller_info[1])])
        self._logger.debug(self._extra_fmt + msg, file_info, tag, *args)

    def info(self, tag, msg, *args):
        """
        Log '[tag] msg % args' with severity 'INFO'.

        Args:
            tag (str): Logger tag.
            msg (str): Logger message.
            args (Any): Auxiliary value.
        """
        caller_info = _find_caller()
        file_info = ':'.join([caller_info[0], str(caller_info[1])])
        self._logger.info(self._extra_fmt + msg, file_info, tag, *args)

    def warn(self, tag, msg, *args):
        """
        Log '[tag] msg % args' with severity 'WARNING'.

        Args:
            tag (str): Logger tag.
            msg (str): Logger message.
            args (Any): Auxiliary value.
        """
        caller_info = _find_caller()
        file_info = ':'.join([caller_info[0], str(caller_info[1])])
        self._logger.warning(self._extra_fmt + msg, file_info, tag, *args)

    def error(self, tag, msg, *args):
        """
        Log '[tag] msg % args' with severity 'ERROR'.

        Args:
            tag (str): Logger tag.
            msg (str): Logger message.
            args (Any): Auxiliary value.
        """
        caller_info = _find_caller()
        file_info = ':'.join([caller_info[0], str(caller_info[1])])
        self._logger.error(self._extra_fmt + msg, file_info, tag, *args)
