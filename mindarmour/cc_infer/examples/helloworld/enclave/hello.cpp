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
#include <string.h>
#include <vector>
#include <iostream>
#include <unordered_map>
#include "helloworld_t.h"
#include "secgear_log.h"
#include "secure_channel_enclave.h"
// #include "switchless_t.h"
#include <time.h>
// #include <openssl/aes.h>


extern "C" {
    #include "tee_log.h"
    #include "tee_time_api.h"
}


#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/types.h"

#include "include/dataset/lite_cv/lite_mat.h"
#include "include/dataset/lite_cv/image_process.h"
#include "include/dataset/vision_lite.h"
#include "include/dataset/execute.h"

using mindspore::dataset::Execute;
using mindspore::dataset::LDataType;
using mindspore::dataset::LiteMat;
using mindspore::dataset::PaddBorderType;
using mindspore::dataset::vision::Decode;
typedef unsigned char Byte;


// void *__dso_handle = (void *) 0;

#define TA_HELLO_WORLD        "secgear hello world!"
#define BUF_MAX 32

char* AESKEY = "df98b715d5c6ed2b25817b6f255411a1";
char* AESIV = "2841ae97419c2973296a0d4bdfe19a4f";    //HEX初始向量

std::unordered_map<std::string, mindspore::Model*> tee_model_map;

std::vector<mindspore::MSTensor> buildTensorFromBuff(std::vector<mindspore::MSTensor> tensors, char *data_buf) {
    int idx = 0;
    for (auto tensor : tensors) {
      auto input_data = tensor.MutableData();
      if (input_data == nullptr) {
        std::cerr << "MallocData for inTensor failed." << std::endl;
      }
      memcpy(reinterpret_cast<char*>(input_data),data_buf+idx,tensor.DataSize());
      idx+=tensor.DataSize();
    }
    return tensors;
}

void buildBufFromTensors(std::vector<mindspore::MSTensor> tensors, char *data_buf) { // possibly add a check on tensor size later
    int idx = 0;
    for(int i=0;i<(int)tensors.size();i++){
        memcpy(data_buf+idx, reinterpret_cast<char*>(tensors[i].MutableData()), tensors[i].DataSize());
        idx+=tensors[i].DataSize();
    }
}

// unsigned char* str2hex(char *str)   
//     {
//         unsigned char *ret = NULL;
//         int str_len = strlen(str);
//         int i = 0;
//         ret = (Byte *)malloc(str_len/2);
//         for (i =0;i < str_len; i = i+2 ) 
//         {
//             sscanf(str+i,"%2hhx",&ret[i/2]);
//         }
//         return ret;
//     }

//     void testEncrytion(char* model_buf, size_t model_size){
//       AES_KEY encryptkey;
//       AES_KEY decryptkey;
//       unsigned char *key;
//       unsigned char *stdiv; 
//       key = str2hex(AESKEY);
//       stdiv = str2hex(AESIV);
//       AES_set_encrypt_key(key,256,&encryptkey);
//       AES_set_decrypt_key(key,256,&decryptkey);

//       size_t pad_len = 16 - model_size%16;
//       size_t padded_len = model_size+pad_len;

//       Byte* plaintext = (Byte *)malloc(padded_len);
//       memcpy(plaintext, reinterpret_cast<Byte*>(model_buf), model_size);
//       memset(plaintext+model_size, 0, pad_len);
      

//       Byte* ciphertxt = (Byte *)malloc(padded_len);
//       memset(ciphertxt, 0, padded_len);
//       unsigned char tmpiv[16];
//       memcpy(tmpiv, stdiv, 16);
//       AES_cbc_encrypt(plaintext, ciphertxt, padded_len, &encryptkey, tmpiv, AES_ENCRYPT);
      
//       // concat IV and ciphertxt
//       size_t pack_size = model_size + 16;
//       Byte* packed_txt = (Byte *)malloc(model_size + 16);
//       memcpy(packed_txt, ciphertxt, model_size);
//       memcpy(packed_txt + model_size, stdiv, 16);

//       // unpack IV and ciphertxt
//       size_t cipher_size = pack_size - 16;
//       size_t deciphertxt_size = cipher_size + 16 - cipher_size%16;
//       Byte* deciphertxt = (Byte *)malloc(deciphertxt_size);
//       memset(deciphertxt, 0, deciphertxt_size);
//       Byte* ciphertxt2 = (Byte *)malloc(deciphertxt_size);
//       memset(deciphertxt, 0, deciphertxt_size);
//       memset(ciphertxt2, 0, deciphertxt_size);
//       memcpy(ciphertxt2, packed_txt, cipher_size);
//       memcpy(tmpiv, packed_txt+cipher_size, 16);

//       AES_cbc_encrypt(ciphertxt2, deciphertxt, deciphertxt_size, &decryptkey, tmpiv, AES_DECRYPT);

//       memcpy(model_buf, deciphertxt, cipher_size);
//     }

// char* getModelDecryptKey(){
//     return AESKEY;
// }

// void decryptModel(char *packed_txt, size_t* pack_size) {
//     // unpack IV and ciphertxt
//     size_t cipher_size = *pack_size - 16;
//     size_t deciphertxt_size = cipher_size + 16 - cipher_size%16;
//     Byte* deciphertxt = (Byte *)malloc(deciphertxt_size);
//     memset(deciphertxt, 0, deciphertxt_size);
//     memcpy(deciphertxt, packed_txt, cipher_size);
//     unsigned char tmpiv[16];
//     memcpy(tmpiv, packed_txt+cipher_size, 16);

//     // get AES key
//     AES_KEY decryptkey;
//     char* aes_key = getModelDecryptKey();
//     unsigned char *key; 
//     key = str2hex(AESKEY);
//     AES_set_decrypt_key(key,256,&decryptkey);

//     // AES_cbc_encrypt(ciphertxt, deciphertxt, padded_len, &decryptkey, tmpiv, AES_DECRYPT);

//     // memcpy(model_buf, deciphertxt, cipher_size);

// }

int enclave_model_build(char *model_buf, size_t model_size, char* model_key, size_t key_size){
    // set context
    auto context = std::make_shared<mindspore::Context>();
    if (context == nullptr) {
        std::cerr << "New context failed." << std::endl;
        return -1;
    }
    // context->SetThreadNum(6);
    // context->SetThreadAffinity({0, 1, 2, 3, 4, 5});
    // context->SetInterOpParallelNum(2);
    auto &device_list = context->MutableDeviceInfo();
    auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
    if (cpu_device_info == nullptr) {
        std::cerr << "New CPUDeviceInfo failed." << std::endl;
        return -1;
    }
    device_list.push_back(cpu_device_info);
    // add decode logic

    // testEncrytion(model_buf, model_size);
    // decryptModel(model_buf, &model_size);

    // build model
    auto tmp_model = new (std::nothrow) mindspore::Model();
    // auto build_ret = tmp_model->Build(model_buf, model_size, mindspore::kMindIR, context);
    unsigned char dec_key[16];
    char* pwd = "1234123412341234";
    memcpy(dec_key, pwd, 16);
    size_t dec_key_len = 16;
    std::string dec_mode = "AES-GCM";
    
    // auto build_ret = tmp_model->Build(model_buf, model_size, mindspore::kMindIR, context, dec_key, dec_key_len, dec_mode);
    auto build_ret = tmp_model->Build(model_buf, model_size, mindspore::kMindIR, context);

    // get model key
    std::string key="";
    for(size_t i=0;i<key_size;i++){
        key.push_back(model_key[i]);
    }

    tee_model_map[key] = tmp_model;
    
    if (build_ret != mindspore::kSuccess) {
        delete tmp_model;
        TEE_LogPrintf("**********Build model error****%s****************\n",build_ret.ToString());
        return -1;
    }
    return 0;

}

int tee_model_get_inputs( char* data_buf, size_t key_size, int* shape_vec){
    // get model key
    std::string key="";
    for(size_t i=0;i<key_size;i++){
        key.push_back(data_buf[i]);
    }
    if(tee_model_map.find(key)==tee_model_map.end()){
        return -1;
    }
    auto model = tee_model_map[key];
    auto inputs = model->GetInputs();
    int idx = 0;
    shape_vec[idx++] = inputs.size();
    for (size_t i=0;i<inputs.size();i++) {
        auto tensor = inputs[i];
        shape_vec[idx++] = tensor.Shape().size();
        for (auto x:tensor.Shape()) {
            shape_vec[idx++] = x;
        }
    }
    buildBufFromTensors(inputs, data_buf);

    return 0;
}

int tee_model_get_outputs( char* data_buf, size_t key_size, int* shape_vec){
    // get model key
    std::string key="";
    for(size_t i=0;i<key_size;i++){
        key.push_back(data_buf[i]);
    }
    if(tee_model_map.find(key)==tee_model_map.end()){
        return -1;
    }
    auto model = tee_model_map[key];
    auto outputs = model->GetOutputs();
    int idx = 0;
    shape_vec[idx++] = outputs.size();
    for (size_t i=0;i<outputs.size();i++) {
        auto tensor = outputs[i];
        shape_vec[idx++] = tensor.Shape().size();
        for (auto x:tensor.Shape()) {
            shape_vec[idx++] = x;
        }
    }
    idx = 0;
    for(int i=0;i<(int)outputs.size();i++){
      memcpy(data_buf+idx, reinterpret_cast<char*>(outputs[i].MutableData()), outputs[i].DataSize());
      idx+=outputs[i].DataSize();
    }

    return 0;
}

int tee_model_dec_predict_enc( size_t key_size, char* data_buf, size_t data_len, bool inputPlaintxt, bool outputPlaintxt, size_t session_id) {
    // decide whether to decrypt or not
    if (!inputPlaintxt) {
        // // decrypt databuf
        // size_t plain_len = 0;
        // auto ret = cc_sec_chl_enclave_decrypt(session_id, data_buf, data_len, nullptr, &plain_len);
        // std::unique_ptr<char[]> plain(new char[plain_len]);
        // ret = cc_sec_chl_enclave_decrypt(session_id, data_buf, data_len, plain.get(), &plain_len);
        // memcpy(data_buf, plain.get(), plain_len);
    }
    
    // get model key
    std::string key="";
    for(size_t i=0;i<key_size;i++){
        key.push_back(data_buf[data_len+i]);
    }
    if(tee_model_map.find(key)==tee_model_map.end()){
        return -1;
    }
    // get model
    auto model = tee_model_map[key];
    // reconstruct input from data_buf
    auto input_tensors = model->GetInputs();
    input_tensors = buildTensorFromBuff(input_tensors, data_buf);
    auto output_tensors = model->GetOutputs();
    auto ret = model->Predict(input_tensors, &output_tensors);
    if (ret != mindspore::kSuccess) {
        return -1;
    }
    int idx = 0;
    size_t output_len = 0;
    for (auto tensor : output_tensors) {
        output_len += tensor.DataSize();
    }
    buildBufFromTensors(output_tensors, data_buf);

    // output encryption
    if (!outputPlaintxt) {
        // char tmp_buf[10000000];
        // memset(tmp_buf, 0, 10000000);
        // size_t encrypt_len = 0;
        // ret = cc_sec_chl_enclave_encrypt(session_id, data_buf, output_len, tmp_buf, &encrypt_len);
        // if (ret != CC_SUCCESS) {
        //     PrintInfo(PRINT_ERROR, "cc_sec_chl_enclave_encrypt failed: %d\n", ret);
        //     return ret;
        // }
        // memset(data_buf, 0, data_len);
        // memcpy(data_buf, &encrypt_len, sizeof(size_t));
        // memcpy(data_buf + sizeof(size_t), tmp_buf, encrypt_len);
    }
    return 0;
}

int tee_model_dec_predict( char* model_key, size_t key_size, char* data_buf, size_t data_len, bool isPlaintxt, size_t session_id, int *shape_vec) {
    // decide whether to decrypt or not
    if (!isPlaintxt) {
        // // decrypt databuf
        // size_t plain_len = 0;
        // auto ret = cc_sec_chl_enclave_decrypt(session_id, data_buf, data_len, nullptr, &plain_len);
        // std::unique_ptr<char[]> plain(new char[plain_len]);
        // ret = cc_sec_chl_enclave_decrypt(session_id, data_buf, data_len, plain.get(), &plain_len);
        // memcpy(data_buf, plain.get(), plain_len);
    }
    
    // get model key
    std::string key="";
    for(size_t i=0;i<key_size;i++){
        key.push_back(model_key[i]);
    }
    if(tee_model_map.find(key)==tee_model_map.end()){
        return -1;
    }
    // get model
    auto model = tee_model_map[key];
    // reconstruct input from data_buf
    auto input_tensors = model->GetInputs();
    input_tensors = buildTensorFromBuff(input_tensors, data_buf);
    auto output_tensors = model->GetOutputs();
    auto ret = model->Predict(input_tensors, &output_tensors);
    if (ret != mindspore::kSuccess) {
        return -1;
    }
    int idx = 0;
    shape_vec[idx++] = output_tensors.size();
    for (size_t i=0;i<output_tensors.size();i++) {
        auto tensor = output_tensors[i];
        shape_vec[idx++] = tensor.Shape().size();
        for (auto x:tensor.Shape()) {
            shape_vec[idx++] = x;
        }
    }
    buildBufFromTensors(output_tensors, data_buf);
    return 0;
}

int tee_model_predict( char* model_key, size_t key_size, char* data_buf, int *shape_vec) {
    // get model key
    std::string key="";
    for(size_t i=0;i<key_size;i++){
        key.push_back(model_key[i]);
    }
    if(tee_model_map.find(key)==tee_model_map.end()){
        return -1;
    }
    // get model
    auto model = tee_model_map[key];
    // reconstruct input from data_buf
    auto input_tensors = model->GetInputs();
    input_tensors = buildTensorFromBuff(input_tensors, data_buf);
    auto output_tensors = model->GetOutputs();
    auto ret = model->Predict(input_tensors, &output_tensors);
    if (ret != mindspore::kSuccess) {
        return -1;
    }
    int idx = 0;
    shape_vec[idx++] = output_tensors.size();
    for (size_t i=0;i<output_tensors.size();i++) {
        auto tensor = output_tensors[i];
        shape_vec[idx++] = tensor.Shape().size();
        for (auto x:tensor.Shape()) {
            shape_vec[idx++] = x;
        }
    }
    buildBufFromTensors(output_tensors, data_buf);
    return 0;
}




