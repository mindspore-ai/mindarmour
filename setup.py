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
setup script
"""
import os
import stat
import shlex
import shutil
import subprocess
from setuptools import find_packages
from setuptools import setup
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py

version = '2.1.0'
cur_dir = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(cur_dir, 'build')


def clean():
    """clean"""
    def readonly_handler(func, path, execinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    if os.path.exists(os.path.join(cur_dir, 'build')):
        shutil.rmtree(os.path.join(cur_dir, 'build'), onerror=readonly_handler)
    if os.path.exists(os.path.join(cur_dir, 'mindarmour.egg-info')):
        shutil.rmtree(os.path.join(cur_dir, 'mindarmour.egg-info'), onerror=readonly_handler)


def write_version(file):
    """write version"""
    file.write("__version__ = '{}'\n".format(version))


def build_depends():
    """generate python file"""
    version_file = os.path.join(cur_dir, 'mindarmour/', 'version.py')
    with open(version_file, 'w') as f:
        write_version(f)


clean()
build_depends()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


def get_description():
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    cmd = "git log --format='[sha1]:%h, [branch]:%d' -1"
    process = subprocess.Popen(
        shlex.split(cmd),
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate()
    if not process.returncode:
        git_version = stdout.decode().strip()
        return "A smart AI security and trustworthy tool box. Git version: %s" % (git_version)
    return "A smart AI security and trustworthy tool box."


class EggInfo(egg_info):
    """Egg info."""
    def run(self):
        super().run()
        egg_info_dir = os.path.join(cur_dir, 'mindarmour.egg-info')
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """BuildPy."""
    def run(self):
        super().run()
        mindarmour_dir = os.path.join(pkg_dir, 'lib', 'mindarmour')
        update_permissions(mindarmour_dir)


setup(
    name='mindarmour',
    version=version,
    author='The MindSpore Authors',
    author_email='contact@mindspore.cn',
    url='https://www.mindspore.cn/',
    download_url='https://gitee.com/mindspore/mindarmour/tags',
    project_urls={
        'Sources': 'https://gitee.com/mindspore/mindarmour',
        'Issue Tracker': 'https://gitee.com/mindspore/mindarmour/issues',
    },
    description=get_description(),
    license='Apache 2.0',
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        'egg_info': EggInfo,
        'build_py': BuildPy,
    },
    install_requires=[
        'scipy >= 1.5.4',
        'numpy >= 1.20.0,<2.0.0',
        'matplotlib >= 3.2.1',
        'pillow >= 9.3.0',
        'scikit-learn >= 0.23.1',
        'easydict >= 1.9',
        'opencv-python >= 4.1.2.30',
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License'
    ]
)
print(find_packages())
