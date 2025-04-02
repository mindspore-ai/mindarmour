/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * secGear is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *     http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
 * PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <stdio.h>
#include <unistd.h>
#include <linux/limits.h>
#include "enclave.h"
#include "helloworld_u.h"
#include "string.h"

#define BUF_LEN 32

int main()
{
    int  retval = 0;
    char *path = PATH;
    char buf[BUF_LEN];
    cc_enclave_t context = {0};
    cc_enclave_result_t res = CC_FAIL;

    char real_p[PATH_MAX];
    /* check file exists, if not exist then use absolute path */
    if (realpath(path, real_p) == NULL) {
        if (getcwd(real_p, sizeof(real_p)) == NULL) {
            printf("Cannot find enclave.sign.so");
            goto end;
        }
        if (PATH_MAX - strlen(real_p) <= strlen("/enclave.signed.so")) {
            printf("Failed to strcat enclave.sign.so path");
            goto end;
        }
        (void)strcat(real_p, "/enclave.signed.so");
    }

    res = cc_enclave_create(real_p, AUTO_ENCLAVE_TYPE, 0, SECGEAR_DEBUG_FLAG, NULL, 0, &context);
    if (res != CC_SUCCESS) {
        printf("host create enclave error\n");
        goto end; 
    }
    printf("host create enclave success\n");

    res = get_string1(&context, &retval, buf);
    if (res != CC_SUCCESS || retval != (int)CC_SUCCESS) {
        printf("Ecall enclave error\n");
    } else {
        printf("enclave say:%s\n", buf);
    }

    res = cc_enclave_destroy(&context);
    if(res != CC_SUCCESS) {
        printf("host destroy enclave error\n");
    } else {
        printf("host destroy enclave success\n");
    }
end:
    return res;
}
