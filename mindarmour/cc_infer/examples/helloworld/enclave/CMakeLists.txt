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

#set sign key
set(PEM Enclave_private.pem)

#set sign tool
set(SIGN_TOOL ${LOCAL_ROOT_PATH}/tools/sign_tool/sign_tool.sh)

#set enclave src code
# set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/hello.c)
aux_source_directory(. SOURCE_FILES)

#set log level
set(PRINT_LEVEL 3)
add_definitions(-DPRINT_LEVEL=${PRINT_LEVEL})

set(MS_DIR "${PROJECT_SOURCE_DIR}/third-party/mindspore-lite-1.10.1-linux-aarch64-tee/runtime")
set(LITECV_DIR "${PROJECT_SOURCE_DIR}/third-party/mindspore-lite-1.10.1-linux-aarch64-tee/runtime/include/dataset")

# set(MS_DIR "${PROJECT_SOURCE_DIR}/third-party/mindspore-lite-1.10.1-linux-aarch64-tee/runtime")
# set(MS_DIR "${PROJECT_SOURCE_DIR}/third-party/mindspore-lite-1.10.1-linux-aarch64/runtime")
# set(LITECV_DIR "${PROJECT_SOURCE_DIR}/third-party/mindspore-lite-1.10.1-linux-aarch64-tee/runtime/include/dataset")

message("MS_DIR: "${MS_DIR})

if(CC_GP)
    #set signed output
    set(OUTPUT ${UUID}.sec)
    #set whilelist. default: /vendor/bin/teec_hello
    set(WHITE_LIST_0 /vendor/bin/helloworld)
    set(WHITE_LIST_OWNER root)
    set(WHITE_LIST_1 /vendor/bin/smliu_infer)
    set(WHITE_LIST_2 /vendor/bin/tee_infer)
    set(WHITELIST WHITE_LIST_0 WHITE_LIST_1 WHITE_LIST_2)

    set(AUTO_FILES  ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_t.h ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_t.c ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_args.h)
    add_custom_command(OUTPUT ${AUTO_FILES}
    DEPENDS ${CURRENT_ROOT_PATH}/${EDL_FILE}
    COMMAND ${CODEGEN} --${CODETYPE} --trusted ${CURRENT_ROOT_PATH}/${EDL_FILE} --search-path ${LOCAL_ROOT_PATH}/inc/host_inc/gp)
endif()

if(CC_SGX)
    set(OUTPUT enclave.signed.so)
    set(AUTO_FILES  ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_t.h ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_t.c)
    add_custom_command(OUTPUT ${AUTO_FILES}
    DEPENDS ${CURRENT_ROOT_PATH}/${EDL_FILE}
    COMMAND ${CODEGEN} --${CODETYPE} --trusted ${CURRENT_ROOT_PATH}/${EDL_FILE} --search-path ${LOCAL_ROOT_PATH}/inc/host_inc/sgx --search-path ${SDK_PATH}/include)
endif()

if(CC_PL)
    set(OUTPUT penglai-ELF)
    set(AUTO_FILES  ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_t.h ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_t.c ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_args.h)
    add_custom_command(OUTPUT ${AUTO_FILES}
    DEPENDS ${CURRENT_ROOT_PATH}/${EDL_FILE}
    COMMAND ${CODEGEN} --${CODETYPE} --trusted ${CURRENT_ROOT_PATH}/${EDL_FILE} --search-path ${LOCAL_ROOT_PATH}/inc/host_inc/penglai)
endif()

set(COMMON_C_FLAGS "-W -Wall -Wno-error  -fno-short-enums  -fno-omit-frame-pointer -fstack-protector \
	-Wstack-protector --param ssp-buffer-size=4 -frecord-gcc-switches -Wextra -nostdinc -nodefaultlibs \
	-fno-peephole -fno-peephole2 -Wno-main -Wno-error=unused-parameter \
	-Wno-error=unused-but-set-variable -Wno-error=format-truncation= -fPIC -fpic")

set(COMMON_CXX_FLAGS "-W -Wno-error -Wno-ignored-qualifiers -fpermissive -fno-short-enums -fno-omit-frame-pointer \
-fstack-protector -Wstack-protector --param ssp-buffer-size=1 -frecord-gcc-switches -Wextra -nostdinc++ -nostdinc \
-nodefaultlibs -fno-use-cxa-atexit -fno-peephole -fno-peephole2 -Wno-main -Wno-error=unused-parameter \
-Wno-error=format-truncation= -Wno-error=implicit-function-declaration -fPIC -fpic -mno-outline-atomics")

#set(COMMON_CXX_FLAGS "-W -Werror -Wno-ignored-qualifiers -fpermissive -fno-short-enums -fno-omit-frame-pointer \
#-fstack-protector -Wstack-protector --param ssp-buffer-size=1 -frecord-gcc-switches -Wextra -nostdinc++ -nostdinc \
#-nodefaultlibs -fno-use-cxa-atexit -fno-peephole -fno-peephole2 -Wno-main -Wno-error=unused-parameter \
#-Wno-error=format-truncation= -Wno-error=implicit-function-declaration -Wno-error=deprecated-copy -fPIC -fpic -mno-outline-atomics")


set(COMMON_C_LINK_FLAGS "-Wl,-z,now -Wl,-z,relro -Wl,-z,noexecstack -Wl,-nostdlib -nodefaultlibs -nostartfiles")

if(CC_GP)
    set(CMAKE_C_FLAGS "${COMMON_C_FLAGS}  -march=armv8-a ")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS}  -s -fPIC")   
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 ${CMAKE_CXX_FLAGS}  -s -fPIC")
    set(CMAKE_SHARED_LINKER_FLAGS "${COMMON_C_LINK_FLAGS} -Wl,-s")
    
    set(HWSECURE_INC ${SDK_PATH}/thirdparty/open_source/libboundscheck/include)
    set(ITRUSTEE_TEEDIR ${SDK_PATH}/)
    set(ITRUSTEE_LIBC ${SDK_PATH}/thirdparty/open_source/musl/libc-liteinfer)
    set(ITRUSTEE_LIBCPP ${SDK_PATH}/include/TA/libc++mslite)
    # set(CCOS_SDK /home/smliu/myccos)
    set(CCOS_SDK ${CMAKE_SOURCE_DIR}/sdk/ccos)
    # set(OpenSSL /home/smliu/openssl_1.1.1k)
    set(OpenSSL ${PROJECT_SOURCE_DIR}/third-party/openssl_1.1.1k)
    # set(ITRUSTRR_SDK_LIB_PATH ${SDK_PATH}/lib/aarch64)

    message("ITRUSTEE_LIBCPP:"${ITRUSTEE_LIBCPP})

    if(${CMAKE_VERSION} VERSION_LESS "3.13.0")
        link_directories(${CMAKE_BINARY_DIR}/lib/        
        )
    endif()

    add_library(${PREFIX} SHARED ${SOURCE_FILES} ${AUTO_FILES})

    target_include_directories( ${PREFIX} PRIVATE
    ${CCOS_SDK} 
    ${CCOS_SDK}/usr/include/c++/7.3.0
    ${CCOS_SDK}/usr/include
    ${CCOS_SDK}/usr/include/c++/7.3.0/include
    ${CCOS_SDK}/usr/include/c++/7.3.0/aarch64-hongmeng-musl
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_BINARY_DIR}/inc
	${LOCAL_ROOT_PATH}/inc/host_inc
	${LOCAL_ROOT_PATH}/inc/host_inc/gp
	${LOCAL_ROOT_PATH}/inc/enclave_inc
	${LOCAL_ROOT_PATH}/inc/enclave_inc/gp
	${ITRUSTEE_TEEDIR}/include/TA
	${ITRUSTEE_TEEDIR}/include/TA/huawei_ext
	${ITRUSTEE_LIBC}/arch/aarch64
	${ITRUSTEE_LIBC}/
	${ITRUSTEE_LIBC}/arch/arm/bits
	${ITRUSTEE_LIBC}/arch/generic
	${ITRUSTEE_LIBC}/arch/arm
	${LOCAL_ROOT_PATH}/inc/enclave_inc/gp/itrustee  
    ${MS_DIR} 
    ${LITECV_DIR}
    ${OpenSSL}/include
    ${CMAKE_SOURCE_DIR}/component/secure_channel/enclave
    )

    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
        target_link_directories(${PREFIX} PRIVATE
	        ${CMAKE_BINARY_DIR}/lib/
            ${ITRUSTEE_TEEDIR}/lib/aarch64
            ${MS_DIR}/lib
            )
    endif()

    foreach(WHITE_LIST ${WHITELIST})
	    add_definitions(-D${WHITE_LIST}="${${WHITE_LIST}}")
    endforeach(WHITE_LIST)
    add_definitions(-DWHITE_LIST_OWNER="${WHITE_LIST_OWNER}")

    target_link_libraries(${PREFIX} 
        secgear_tee.a
        -Wl,--whole-archive ${MS_DIR}/lib/libmindspore-lite.a -Wl,--no-whole-archive
        ${MS_DIR}/lib/libminddata-lite.a
        #-Wl,--whole-archive ${MS_DIR}/lib/libmindspore-lite.a -Wl,--no-whole-archive
        ${PROJECT_SOURCE_DIR}/third-party/libs/libclang_rt.builtins-aarch64.a
        ${PROJECT_SOURCE_DIR}/third-party/libs/libstdc++.a
        # /home/smliu/infer_secgear/mylib/libstdc++.a
        ${MS_DIR}/third_party/libjpeg-turbo/lib/libjpeg.a
        ${MS_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.a
        # /home/smliu/openssl_1.1.1k/lib/libcrypto.a
        unwind.a gcc.a 
    )

    #for trustzone compiling, you should connact us to get config and private_key.pem for test, so we will not sign and install binary in this example #
        add_custom_command(TARGET ${PREFIX}
    	      POST_BUILD
	      	      COMMAND bash ${SIGN_TOOL} -d sign -x trustzone -i ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${PREFIX}.so -c ${CMAKE_CURRENT_SOURCE_DIR}/manifest.txt -m ${CMAKE_CURRENT_SOURCE_DIR}/config_cloud.ini -o ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${OUTPUT})

          install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${OUTPUT}  
              DESTINATION /data
              PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE  WORLD_READ  WORLD_EXECUTE)

endif()

if(CC_SGX)
    set(SGX_DIR ${SDK_PATH})
    set(CMAKE_C_FLAGS "${COMMON_C_FLAGS} -m64 -fvisibility=hidden")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS}  -s")
    set(LINK_LIBRARY_PATH ${SGX_DIR}/lib64)

    if(CC_SIM)
        set(Trts_Library_Name sgx_trts_sim)
        set(Service_Library_Name sgx_tservice_sim)
    else()
        set(Trts_Library_Name sgx_trts)
        set(Service_Library_Name sgx_tservice)
    endif()

    set(Crypto_Library_Name sgx_tcrypto)

    set(CMAKE_SHARED_LINKER_FLAGS  "${COMMON_C_LINK_FLAGS} -Wl,-z,defs -Wl,-pie -Bstatic -Bsymbolic -eenclave_entry \
	-Wl,--export-dynamic -Wl,--defsym,__ImageBase=0 -Wl,--gc-sections -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/Enclave.lds")

    if(${CMAKE_VERSION} VERSION_LESS "3.13.0") 
        link_directories(${LINK_LIBRARY_PATH})
    endif()

    add_library(${PREFIX}  SHARED ${SOURCE_FILES} ${AUTO_FILES})

    target_include_directories(${PREFIX} PRIVATE
	    ${CMAKE_CURRENT_BINARY_DIR}
	    ${SGX_DIR}/include/tlibc
	    ${SGX_DIR}/include/libcxx
	    ${SGX_DIR}/include
	    ${LOCAL_ROOT_PATH}/inc/host_inc
	    ${LOCAL_ROOT_PATH}/inc/host_inc/sgx)
    
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0") 
        target_link_directories(${PREFIX} PRIVATE
	        ${LINK_LIBRARY_PATH})
    endif()

    target_link_libraries(${PREFIX}  -Wl,--whole-archive ${Trts_Library_Name} -Wl,--no-whole-archive 
    		-Wl,--start-group -lsgx_tstdc -lsgx_tcxx -l${Crypto_Library_Name} -l${Service_Library_Name}   -Wl,--end-group)
    add_custom_command(TARGET ${PREFIX} 
    POST_BUILD
    COMMAND umask 0177
    COMMAND openssl genrsa -3 -out ${PEM} 3072
    COMMAND bash ${SIGN_TOOL} -d sign -x sgx -i ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${PREFIX}.so -k ${PEM} -o ${OUTPUT} -c ${CMAKE_CURRENT_SOURCE_DIR}/Enclave.config.xml)
endif()

if(NOT DEFINED CC_PL)
    set_target_properties(${PREFIX} PROPERTIES SKIP_BUILD_RPATH TRUE)
endif()

if(CC_PL)
    set(SDK_LIB_DIR ${SDK_PATH}/lib)
    set(SDK_INCLUDE_DIR ${SDK_LIB_DIR}/app/include)
    set(SDK_APP_LIB ${SDK_LIB_DIR}/libpenglai-enclave-eapp.a)
    set(MUSL_LIB_DIR ${SDK_PATH}/musl/lib)
    set(MUSL_LIBC ${MUSL_LIB_DIR}/libc.a)
    set(GCC_LIB ${SDK_LIB_DIR}/libgcc.a)
    set(SECGEAR_TEE_LIB ${CMAKE_BINARY_DIR}/lib/libsecgear_tee.a)

    set(SOURCE_C_OBJS "")
    foreach(SOURCE_FILE ${SOURCE_FILES})
        STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" SOURCE_FILE_NAME ${SOURCE_FILE})
        set(SOURCE_OBJ ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_FILE_NAME}.o)
        add_custom_command(
            OUTPUT ${SOURCE_OBJ}
            DEPENDS ${SOURCE_FILES}
            COMMAND gcc -Wall -I${SDK_INCLUDE_DIR} -I${CMAKE_CURRENT_BINARY_DIR} -I${CMAKE_BINARY_DIR}/inc
                -I${LOCAL_ROOT_PATH}/inc/host_inc -I${LOCAL_ROOT_PATH}/inc/host_inc/penglai -I${LOCAL_ROOT_PATH}/inc/enclave_inc
                -I${LOCAL_ROOT_PATH}/inc/enclave_inc/penglai -c -o ${SOURCE_OBJ} ${SOURCE_FILE}
            COMMENT "generate SOURCE_OBJ"
        )
        list(APPEND SOURCE_C_OBJS ${SOURCE_OBJ})
    endforeach()

    set(APP_C_OBJ ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_t.o)
    add_custom_command(
        OUTPUT ${APP_C_OBJ}
        DEPENDS ${AUTO_FILES}
        COMMAND gcc -Wall -I${SDK_INCLUDE_DIR} -I${CMAKE_CURRENT_BINARY_DIR} -I${CMAKE_BINARY_DIR}/inc
            -I${LOCAL_ROOT_PATH}/inc/host_inc -I${LOCAL_ROOT_PATH}/inc/host_inc/penglai -I${LOCAL_ROOT_PATH}/inc/enclave_inc
            -I${LOCAL_ROOT_PATH}/inc/enclave_inc/penglai -c -o ${APP_C_OBJ} ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}_t.c
        COMMENT "generate APP_C_OBJ"
    )

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${OUTPUT}
        DEPENDS ${APP_C_OBJ} ${SOURCE_C_OBJS} ${SDK_APP_LIB} ${MUSL_LIBC} ${GCC_LIB}
        COMMAND ld -static -L${SDK_LIB_DIR} -L${MUSL_LIB_DIR} -L/usr/lib64 -lpenglai-enclave-eapp -lsecgear_tee -lc
            -o ${CMAKE_CURRENT_SOURCE_DIR}/${OUTPUT} ${APP_C_OBJ} ${SOURCE_C_OBJS} ${SDK_APP_LIB} ${SECGEAR_TEE_LIB}
            ${MUSL_LIBC} ${GCC_LIB} -T ${SDK_PATH}/app.lds
        COMMAND chmod -x ${CMAKE_CURRENT_SOURCE_DIR}/${OUTPUT}
        COMMENT "generate penglai-ELF"
    )
    add_custom_target(
        ${OUTPUT} ALL
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${OUTPUT}
        COMMENT "makefile target penglai-ELF"
    )

endif()




