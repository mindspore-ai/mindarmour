# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
# secGear is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.

project(HelloWorld CXX C)

# gcc version 9.4.0
set(CMAKE_C_COMPILER "/home/hanzhibin/packages/gcc_build/output/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/hanzhibin/packages/gcc_build/output/bin/g++")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

set(CURRENT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

#set edl name
set(EDL_FILE helloworld.edl)
set(CODEGEN codegen)

if(CC_GP)
    set(CODETYPE trustzone)
    # set(UUID 2f878f0e-a8b7-4a4f-9c6c-52f1cd321252)
    set(UUID 318bf1bf-b5b2-460e-81b6-bbb4c25fd545)
    add_definitions(-DPATH="/data/${UUID}.sec")
endif()

if(CC_SGX)
    set(CODETYPE sgx)
    add_definitions(-DPATH="${CMAKE_CURRENT_BINARY_DIR}/enclave/enclave.signed.so")
endif()

if(CC_PL)
    set(CODETYPE penglai)
    add_definitions(-DPATH="${CMAKE_CURRENT_SOURCE_DIR}/enclave/penglai-ELF")
endif()

add_subdirectory(${CURRENT_ROOT_PATH}/enclave)
add_subdirectory(${CURRENT_ROOT_PATH}/host)
