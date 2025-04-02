# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
# secGear is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.

#set auto code prefix
set(PREFIX helloworld)
#set host exec name
set(OUTPUT tee_infer)
#set host src code
set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

set(MS_DIR "${PROJECT_SOURCE_DIR}/third-party/mindspore-lite-1.10.1-linux-aarch64-ree/runtime")
set(LITECV_DIR "${PROJECT_SOURCE_DIR}/third-party/mindspore-lite-1.10.1-linux-aarch64-ree/runtime/include/dataset")

#set(MS_DIR "${PROJECT_SOURCE_DIR}/third-party/mindspore-lite-2.1.0-linux-aarch64/runtime")
#set(LITECV_DIR "${PROJECT_SOURCE_DIR}/third-party/mindspore-lite-2.1.0-linux-aarch64/runtime/include/dataset")

message("MS_DIR HOST: "${MS_DIR})

#set auto code
if(CC_GP)
    set(AUTO_FILES  ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_u.h ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_u.c ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_args.h)
    add_custom_command(OUTPUT ${AUTO_FILES}
    DEPENDS ${CURRENT_ROOT_PATH}/${EDL_FILE}
    COMMAND ${CODEGEN} --${CODETYPE} --untrusted ${CURRENT_ROOT_PATH}/${EDL_FILE} --search-path ${LOCAL_ROOT_PATH}/inc/host_inc/gp)
endif()
    
if(CC_SGX)
    set(AUTO_FILES  ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_u.h ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_u.c)
    add_custom_command(OUTPUT ${AUTO_FILES}
    DEPENDS ${CURRENT_ROOT_PATH}/${EDL_FILE}
    COMMAND ${CODEGEN} --${CODETYPE} --untrusted ${CURRENT_ROOT_PATH}/${EDL_FILE} --search-path ${LOCAL_ROOT_PATH}/inc/host_inc/sgx  --search-path ${SDK_PATH}/include)
endif()

if(CC_PL)
    set(AUTO_FILES  ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_u.h ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_u.c)
    add_custom_command(OUTPUT ${AUTO_FILES}
    DEPENDS ${CURRENT_ROOT_PATH}/${EDL_FILE}
    COMMAND ${CODEGEN} --${CODETYPE} --untrusted ${CURRENT_ROOT_PATH}/${EDL_FILE} --search-path ${LOCAL_ROOT_PATH}/inc/host_inc/penglai)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIE  -Wno-error")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS}  -s")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -fPIC  -Wno-error")

if(CC_GP)
    if(${CMAKE_VERSION} VERSION_LESS "3.13.0")
        link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    endif()
    add_executable(${OUTPUT} ${SOURCE_FILE} ${AUTO_FILES})
    target_include_directories(${OUTPUT} PRIVATE
            ${CMAKE_BINARY_DIR}/inc
            ${LOCAL_ROOT_PATH}/inc/host_inc
            ${LOCAL_ROOT_PATH}/inc/host_inc/gp
            ${CMAKE_CURRENT_BINARY_DIR}
            ${MS_DIR} 
            ${LITECV_DIR}
            )
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
        target_link_directories(${OUTPUT} PRIVATE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        ${MS_DIR}/lib
        ${MS_DIR}/third_party/libjpeg-turbo/lib
        )
    endif()

    target_link_libraries(${OUTPUT} 
    
    -Wl,--whole-archive ${MS_DIR}/lib/libmindspore-lite.a -Wl,--no-whole-archive
    ${MS_DIR}/lib/libminddata-lite.a
    ${MS_DIR}/third_party/securec/libsecurec.a
    ${PROJECT_SOURCE_DIR}/third-party/openssl_1.1.1k/lib/libcrypto.a
    # /home/smliu/openssl_1.1.1k/lib/libcrypto.a
    # ${MS_DIR}/third_party/libjpeg-turbo/lib/libjpeg.a
    # ${MS_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.a
    )
endif()

if(CC_SGX)
    if(${CMAKE_VERSION} VERSION_LESS "3.13.0")
        link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    endif()
    add_executable(${OUTPUT} ${SOURCE_FILE} ${AUTO_FILES})
    target_include_directories(${OUTPUT} PRIVATE
			${LOCAL_ROOT_PATH}/inc/host_inc 
			${LOCAL_ROOT_PATH}/inc/host_inc/sgx
			${CMAKE_CURRENT_BINARY_DIR})
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
        target_link_directories(${OUTPUT} PRIVATE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    endif()
endif()

if(CC_PL)
    if(${CMAKE_VERSION} VERSION_LESS "3.13.0")
        link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    endif()
    add_executable(${OUTPUT} ${SOURCE_FILE} ${AUTO_FILES})
    target_include_directories(${OUTPUT} PRIVATE
			${LOCAL_ROOT_PATH}/inc/host_inc
			${LOCAL_ROOT_PATH}/inc/host_inc/penglai
			${CMAKE_CURRENT_BINARY_DIR})
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
        target_link_directories(${OUTPUT} PRIVATE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    endif()
endif()

if(CC_SIM)
    target_link_libraries(${OUTPUT} secgearsim pthread)
else()
    target_link_libraries(${OUTPUT} secgear pthread)
endif()
set_target_properties(${OUTPUT} PROPERTIES SKIP_BUILD_RPATH TRUE)

if(CC_GP)
	#itrustee install whitelist /vender/bin/teec_hello 
	install(TARGETS  ${OUTPUT}
        	RUNTIME
        	DESTINATION /vendor/bin/
       	 	PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
		            GROUP_EXECUTE GROUP_READ
		            WORLD_EXECUTE WORLD_READ)  
endif()

if(CC_SGX)
	install(TARGETS  ${OUTPUT}
        	RUNTIME
        	DESTINATION ${CMAKE_BINARY_DIR}/bin/
       	 	PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
		            GROUP_EXECUTE GROUP_READ
		            WORLD_EXECUTE WORLD_READ)  
endif()

if(CC_PL)
	install(TARGETS  ${OUTPUT}
		RUNTIME
		DESTINATION ${CMAKE_BINARY_DIR}/bin/
       	 	PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
		            GROUP_EXECUTE GROUP_READ
		            WORLD_EXECUTE WORLD_READ)  
endif()

