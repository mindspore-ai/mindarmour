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

#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <random>
#include <time.h>
#include <chrono>

#include "secgear_uswitchless.h"
#include "secgear_shared_memory.h"

#define BUF_LEN 32

#include "include/api/types.h"
#include "include/dataset/lite_cv/lite_mat.h"
#include "include/dataset/lite_cv/image_process.h"
#include "include/dataset/vision_lite.h"
#include "include/dataset/execute.h"

#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/status.h"

#include <openssl/aes.h>

using mindspore::dataset::Execute;
using mindspore::dataset::LDataType;
using mindspore::dataset::LiteMat;
using mindspore::dataset::PaddBorderType;
using mindspore::dataset::vision::Decode;

using ShapeValueDType = int64_t;
using ShapeVector = std::vector<ShapeValueDType>;
typedef unsigned char Byte;

char* AESKEY = "df98b715d5c6ed2b25817b6f255411a1";   //HEX密钥
char* AESIV = "2841ae97419c2973296a0d4bdfe19a4f";    //HEX初始向量

constexpr int kIndex0 = 0;
constexpr int kIndex1 = 1;
constexpr int kIndex2 = 2;
constexpr int kIndex3 = 3;

template <class T>
void TransposeDim4(const ShapeVector &input_shape, const ShapeVector &output_shape, const std::vector<int> &perm,
                   const T *const in_data, T *out_data) {
  auto num_axes = input_shape.size();
  std::vector<int64_t> strides;
  std::vector<int64_t> out_strides;
  strides.resize(num_axes);
  out_strides.resize(num_axes);
  strides[num_axes - 1] = 1LL;
  out_strides[num_axes - 1] = 1LL;
  for (size_t i = num_axes - 1; i >= 1; i--) {
    strides[i - 1] = input_shape[i] * strides[i];
    out_strides[i - 1] = output_shape[i] * out_strides[i];
  }
  const auto stride0 = strides[perm[kIndex0]];
  const auto stride1 = strides[perm[kIndex1]];
  const auto stride2 = strides[perm[kIndex2]];
  const auto stride3 = strides[perm[kIndex3]];
  const auto out_stride0 = out_strides[kIndex0];
  const auto out_stride1 = out_strides[kIndex1];
  const auto out_stride2 = out_strides[kIndex2];
  const auto output0 = output_shape[kIndex0];
  const auto output1 = output_shape[kIndex1];
  const auto output2 = output_shape[kIndex2];
  const auto output3 = output_shape[kIndex3];

  int64_t out_beg_i = 0;
  int64_t beg_i = 0;
  for (int64_t i = 0; i < output0; ++i) {
    int64_t out_beg_ij = out_beg_i;
    int64_t beg_ij = beg_i;
    for (int64_t j = 0; j < output1; ++j) {
      int64_t out_beg_ijk = out_beg_ij;
      int64_t beg_ijk = beg_ij;
      for (int64_t k = 0; k < output2; ++k) {
        for (int64_t m = 0; m < output3; ++m) {
          out_data[out_beg_ijk + m] = in_data[beg_ijk + m * stride3];
        }
        out_beg_ijk += out_stride2;
        beg_ijk += stride2;
      }
      out_beg_ij += out_stride1;
      beg_ij += stride1;
    }
    out_beg_i += out_stride0;
    beg_i += stride0;
  }
}

template <typename T>
int DoTransposeData(mindspore::MSTensor tensor, std::vector<int> perm) {//NCHW->NHWC 0123->0231, NHWC2NHCW 0132
  auto origin_shape = tensor.Shape();
  ShapeVector new_shape;
  for (auto &val : perm) {
    new_shape.push_back(origin_shape[val]);
  }
  int64_t count = 1;
  for (const auto &dat : origin_shape) {
    count *= dat;
  }
  std::vector<T> buf(count);
  void *originData = tensor.MutableData();
  T *inputdata = static_cast<T *>(originData);
  TransposeDim4<T>(origin_shape, new_shape, perm, inputdata, buf.data());
  // memcpy_s(tensor.MutableData(), tensor.DataSize(), buf.data(), count * sizeof(T));
  memcpy(tensor.MutableData(), buf.data(), tensor.DataSize());
  tensor.SetShape(new_shape);
  return 0;
}

std::vector<mindspore::MSTensor> NCHW2NHWC(std::vector<mindspore::MSTensor> tensors) {
  for(auto &tensor:tensors){
    DoTransposeData<float>(tensor, std::vector<int>{{0,2,3,1}});
  }
  return tensors;
}

std::vector<mindspore::MSTensor> NHWC2NCHW(std::vector<mindspore::MSTensor> tensors) {
  for(auto &tensor:tensors){
    DoTransposeData<float>(tensor, std::vector<int>{{0,3,1,2}});
  }
  return tensors;
}

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
}

int GenerateInputDataWithRandom(std::vector<mindspore::MSTensor> inputs) {
  for (auto tensor : inputs) {
    auto input_data = tensor.MutableData();
    if (input_data == nullptr) {
      std::cerr << "MallocData for inTensor failed." << std::endl;
      return -1;
    }
    // for float input data
    // GenerateRandomData<float>(tensor.DataSize(), input_data, std::uniform_real_distribution<float>(0.1f, 1.0f));
    // for int input data
    GenerateRandomData<int32_t>(tensor.DataSize(), input_data, std::uniform_int_distribution<int32_t>(1, 128));
  }
  return mindspore::kSuccess;
}

char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    std::cerr << "file is nullptr." << std::endl;
    return nullptr;
  }

  std::ifstream ifs(file, std::ifstream::in | std::ifstream::binary);
  if (!ifs.good()) {
    std::cerr << "file: " << file << " is not exist." << std::endl;
    return nullptr;
  }

  if (!ifs.is_open()) {
    std::cerr << "file: " << file << " open failed." << std::endl;
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char[]> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    std::cerr << "malloc buf failed, file: " << file << std::endl;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}

void dumpTensor(std::vector<mindspore::MSTensor> tensors, std::string filename, std::vector<int> perm){
  size_t size = 0;
  for(auto tensor:tensors){
    DoTransposeData<float>(tensor, perm);
    size += tensor.DataSize();
  }

  int idx = 0;
  std::vector<char> buf(size);
    
  for(int i=0;i<(int)tensors.size();i++){
    memcpy(buf.data()+idx, reinterpret_cast<char*>(tensors[i].MutableData()), tensors[i].DataSize());
    idx+=tensors[i].DataSize();
  }

  std::ofstream saveFile(filename,std::ifstream::binary);
  saveFile.write(buf.data(),size);
}

class CCModel {
  public:
    std::vector<std::string> model_paths;
    std::vector<std::string> backend_lists;
    size_t num_models;
    std::vector<char *> model_bufs;
    std::vector<size_t> model_buf_sizes;
    std::unordered_map<std::string, mindspore::Model*> ree_model_map;
    char *key_buf;



    CCModel(std::vector<std::string> model_paths, std::vector<std::string> backend_lists) {
      this->model_paths = model_paths;
      this->backend_lists = backend_lists;
      this->num_models = model_paths.size();
      std::cout<<"There are "<< num_models << " models to build."<<std::endl;
      this->Init();
    }

    int Init(){
      this->model_bufs.resize(num_models);
      this->model_buf_sizes.resize(num_models);
      for (size_t i=0;i<num_models;i++) {
        this->model_bufs[i] = ReadFile(this->model_paths[i].c_str(), &this->model_buf_sizes[i]);
      }
      return 0;
    }

    ~CCModel() {
      // auto ret = cc_free_shared_memory(&context, key_buf);
      // if(ret!=0){
      //   std::cerr << "cc_free_shared_memory failed." << std::endl;
      // }
    }

    int Build(cc_enclave_t context, int* retval){  
          
      for(size_t i=0;i<num_models;i++) {
        if(backend_lists[i]=="TEE_CPU") {
          auto build_ret = buildInCPUTEE(this->model_bufs[i], this->model_buf_sizes[i], this->model_paths[i], context, retval);
          if(build_ret!=0){
            std::cerr<<"buildInCPUTEE ERROR!"<<std::endl;
            return -1;
          }
        } else if (backend_lists[i]=="REE_CPU_D") {
          auto build_ret = buildInCPUREE2(this->model_bufs[i], this->model_buf_sizes[i], this->model_paths[i]);
          if(build_ret!=0){
            std::cerr<<"buildInCPUREE ERROR!"<<std::endl;
            return -1;
          }
        } else if (backend_lists[i]=="REE_CPU") {
          auto build_ret = buildInCPUREE(this->model_bufs[i], this->model_buf_sizes[i], this->model_paths[i]);
          if(build_ret!=0){
            std::cerr<<"buildInCPUREE ERROR!"<<std::endl;
            return -1;
          }
        } else if (backend_lists[i]=="REE_NPU") {
          auto build_ret = buildInNPUREE(this->model_bufs[i], this->model_buf_sizes[i], this->model_paths[i]);
          if(build_ret!=0){
            std::cerr<<"buildInNPUREE ERROR!"<<std::endl;
            return -1;
          }
        }
      }
      return 0;
    }

    void model_buf_encrypt(char* model_buf, size_t model_size){

    }

    int buildInCPUTEE( char* model_buf, size_t model_size, std::string model_key, cc_enclave_t context, int *retval) {
      std::cout<<"start building in CPU TEE"<<std::endl;
      char *shared_buf = (char *)cc_malloc_shared_memory(&context, model_size);
      memcpy(shared_buf, model_buf, model_size);

      size_t key_size = model_key.length();
      // char key_buf[key_size];
      char *key_buf = (char *)cc_malloc_shared_memory(&context, key_size);
      memcpy(key_buf,model_key.c_str(),key_size);

      auto ret = enclave_model_build(&context, retval, shared_buf, model_size, key_buf, key_size);

      ret = cc_free_shared_memory(&context, shared_buf);
      ret = cc_free_shared_memory(&context, key_buf);
      if(ret!=0){
        std::cerr << "enclave_model_build failed." << std::endl;
        return -1;
      }
      return 0;
    }

    unsigned char* str2hex(char *str)   
    {
        unsigned char *ret = NULL;
        int str_len = strlen(str);
        int i = 0;
        ret = (Byte *)malloc(str_len/2);
        for (i =0;i < str_len; i = i+2 ) 
        {
            sscanf(str+i,"%2hhx",&ret[i/2]);
        }
        return ret;
    }

    void testEncrytion(char* model_buf, size_t model_size){
      printf("model_buf: ");
      for(size_t i = 0; i< model_size; i++)
      {
          std::cout<< (int)model_buf[i] <<" ";
      }
      printf("\n" );

      AES_KEY encryptkey;
      AES_KEY decryptkey;
      unsigned char *key;
      unsigned char *stdiv; 
      key = str2hex(AESKEY);
      stdiv = str2hex(AESIV);
      AES_set_encrypt_key(key,256,&encryptkey);
      AES_set_decrypt_key(key,256,&decryptkey);



      size_t pad_len = 16 - model_size%16;
      size_t padded_len = model_size+pad_len;

      Byte* plaintext = (Byte *)malloc(padded_len);
      memcpy(plaintext, reinterpret_cast<Byte*>(model_buf), model_size);
      memset(plaintext+model_size, 0, pad_len);
      printf("plain_text: len = %d\n", model_size);

      for(size_t i = 0; i< padded_len; i++)
      {
          printf("%02X ", plaintext[i]);
      }
      printf("\n" );
      

      Byte* ciphertxt = (Byte *)malloc(padded_len);
      memset(ciphertxt, 0, padded_len);
      unsigned char tmpiv[16];
      memcpy(tmpiv, stdiv, 16);
      AES_cbc_encrypt(plaintext, ciphertxt, padded_len, &encryptkey, tmpiv, AES_ENCRYPT);
      
      printf("encrypted_text: " );
      for(int i = 0; i < padded_len; i++)
      {
          printf("%02X ", ciphertxt[i]);
      }
      printf("\n" );

      // concat IV and ciphertxt
      size_t pack_size = padded_len + 16;
      Byte* packed_txt = (Byte *)malloc(pack_size);
      memcpy(packed_txt, ciphertxt, padded_len);
      memcpy(packed_txt + padded_len, stdiv, 16);
      printf("packed_txt: " );
      for(int i = 0; i < pack_size; i++)
      {
          printf("%02X ", packed_txt[i]);
      }
      printf("\n" );

      // unpack IV and ciphertxt
      size_t deciphertxt_size = pack_size - 16;
      // size_t deciphertxt_size = cipher_size + 16 - cipher_size%16;

      Byte* deciphertxt = (Byte *)malloc(deciphertxt_size);
      memset(deciphertxt, 0, deciphertxt_size);

      Byte* ciphertxt2 = (Byte *)malloc(deciphertxt_size);
      memset(ciphertxt2, 0, deciphertxt_size);

      memcpy(ciphertxt2, packed_txt, deciphertxt_size);
      printf("ciphertxt2: " );
      for(int i = 0; i < deciphertxt_size; i++)
      {
          printf("%02X ", ciphertxt2[i]);
      }
      printf("\n" );
      memcpy(tmpiv, packed_txt+deciphertxt_size, 16);
      // memcpy(tmpiv, stdiv, 16);

      // AES_cbc_encrypt(ciphertxt, deciphertxt, padded_len, &decryptkey, tmpiv, AES_DECRYPT);
      AES_cbc_encrypt(ciphertxt2, deciphertxt, deciphertxt_size, &decryptkey, tmpiv, AES_DECRYPT);
      printf("decrypted_text: " );
      for(int i = 0; i < padded_len; i++)
      {
          printf("%02X ", deciphertxt[i]);
      }
      printf("\n" );
      // for(int i = 0; i < padded_len; i++)
      // {
      //     printf("%d ", deciphertxt[i]);
      // }
      // printf("\n" );
      // memset(model_buf, 0, cipher_size);
      // memcpy(model_buf, deciphertxt, cipher_size);
    }



    int buildInCPUREE(char* model_buf, size_t model_size, std::string model_key) {
      std::cout<<"start building in CPU REE"<<std::endl;
      auto context = std::make_shared<mindspore::Context>();
      if (context == nullptr) {
          std::cerr << "New context failed." << std::endl;
          return -1;
      }
      auto &device_list = context->MutableDeviceInfo();
      auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
      if (cpu_device_info == nullptr) {
          std::cerr << "New CPUDeviceInfo failed." << std::endl;
          return -1;
      }
      device_list.push_back(cpu_device_info);
      std::cout<<"finish CPU REE context initialization."<<std::endl;

      auto tmp_model = new (std::nothrow) mindspore::Model();

      // testEncrytion("123456", 6);
      // testEncrytion(model_buf, model_size);
      
      auto build_ret = tmp_model->Build(model_buf, model_size, mindspore::kMindIR, context);
      std::cout<<"finish CPU REE build."<<std::endl;
      this->ree_model_map[model_key] = tmp_model;
      std::cout<<"stored CPU REE model."<<std::endl;
      
      if (build_ret != mindspore::kSuccess) {
          delete tmp_model;
          std::cerr << "build "<< model_key <<" failed." << std::endl;
          return -1;
      }

      return 0;
    }

    int buildInCPUREE2(char* model_buf, size_t model_size, std::string model_key) {
      std::cout<<"start building in CPU REE with decrytion"<<std::endl;
      auto context = std::make_shared<mindspore::Context>();
      if (context == nullptr) {
          std::cerr << "New context failed." << std::endl;
          return -1;
      }
      auto &device_list = context->MutableDeviceInfo();
      auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
      if (cpu_device_info == nullptr) {
          std::cerr << "New CPUDeviceInfo failed." << std::endl;
          return -1;
      }
      device_list.push_back(cpu_device_info);
      std::cout<<"finish CPU REE context initialization."<<std::endl;

      auto tmp_model = new (std::nothrow) mindspore::Model();

      // testEncrytion("123456", 6);
      // testEncrytion(model_buf, model_size);

      unsigned char dec_key[16];
      char* pwd = "1234123412341234";
      memcpy(dec_key, pwd, 16);
      size_t dec_key_len = 16;
      std::string dec_mode = "AES-GCM";
      
      auto build_ret = tmp_model->Build(model_buf, model_size, mindspore::kMindIR, context, dec_key, dec_key_len, dec_mode);
      std::cout<<"finish CPU REE build."<<std::endl;
      this->ree_model_map[model_key] = tmp_model;
      std::cout<<"stored CPU REE model."<<std::endl;
      
      if (build_ret != mindspore::kSuccess) {
          delete tmp_model;
          std::cerr << "build "<< model_key <<" failed." << std::endl;
          return -1;
      }

      return 0;
    }

    int buildInNPUREE(char* model_buf, size_t model_size, std::string model_key) {
      std::cout<<"start building in NPU REE"<<std::endl;
      auto context = std::make_shared<mindspore::Context>();
      if (context == nullptr) {
          std::cerr << "New context failed." << std::endl;
          return -1;
      }
      context->SetThreadNum(6);
      context->SetThreadAffinity({0, 1, 2, 3, 4, 5});
      context->SetInterOpParallelNum(2);
      auto &device_list = context->MutableDeviceInfo();
      auto npu_device_info = std::make_shared<mindspore::AscendDeviceInfo>();
      if (npu_device_info == nullptr) {
          std::cerr << "New NPUDeviceInfo failed." << std::endl;
          return -1;
      }
      int32_t device_id = 0;
      // Set Ascend 310/310P/910 device id.
      npu_device_info->SetDeviceID(device_id);
      // The Ascend device context needs to be push_back into device_list to work.
      device_list.push_back(npu_device_info);

      auto tmp_model = new (std::nothrow) mindspore::Model();
      auto build_ret = tmp_model->Build(model_buf, model_size, mindspore::kMindIR, context);

      this->ree_model_map[model_key] = tmp_model;
      
      if (build_ret != mindspore::kSuccess) {
          delete tmp_model;
          std::cerr << "build "<< model_key <<" failed." << std::endl;
          return -1;
      }

      return 0;
    }

    std::vector<mindspore::MSTensor> buildTensorFromBuff(int *shape_vec, char *data_buf) {
      std::vector<mindspore::MSTensor> tensors;
      std::cout<<"num tensors = "<<shape_vec[0]<<std::endl;
      int idx = 1;
      int data_buf_idx = 0;
      for (int i=0;i<shape_vec[0];i++){
        std::cout<<"tensor "<<i+1 <<": ";
        int num_dim = shape_vec[idx++];
        std::cout<<num_dim<<" ";
        ShapeVector tensor_shape;
        tensor_shape.resize(num_dim);
        size_t datalen = 4;
        for (int j=0;j<num_dim;j++) {
          tensor_shape[j] = shape_vec[idx++];
          datalen *= tensor_shape[j];
          std::cout<< tensor_shape[j] <<" ";
        }
        std::cout<<std::endl;
        mindspore::MSTensor tmp_tensor("tensor", mindspore::DataType::kNumberTypeFloat32,tensor_shape,data_buf+data_buf_idx,datalen);
        tensors.emplace_back(tmp_tensor);
      }
      return tensors;
    }

    void buildBufFromTensors(std::vector<mindspore::MSTensor> tensors, char *data_buf, size_t* data_len) { // possibly add a check on tensor size later
      size_t idx = 0;
      for(int i=0;i<(int)tensors.size();i++){
          memcpy(data_buf+idx, reinterpret_cast<char*>(tensors[i].MutableData()), tensors[i].DataSize());
          idx+=tensors[i].DataSize();
      }
      *data_len = idx;
    }

    void buildBufFromTensors(std::vector<mindspore::MSTensor> tensors, char *data_buf) { // possibly add a check on tensor size later
      size_t idx = 0;
      for(int i=0;i<(int)tensors.size();i++){
          memcpy(data_buf+idx, reinterpret_cast<char*>(tensors[i].MutableData()), tensors[i].DataSize());
          idx+=tensors[i].DataSize();
      }
    }

    std::vector<mindspore::MSTensor> GetInputsOf(int i, cc_enclave_t context, int *retval ){
      std::vector<mindspore::MSTensor> inputs;
      if(this->backend_lists[i]=="TEE_CPU"){
        size_t key_size = this->model_paths[i].length();
        char *data_buf = (char *)cc_malloc_shared_memory(&context, 10000000);
        memcpy(data_buf,this->model_paths[i].c_str(),key_size);
        // set shape vec
        int *shape_vec = (int *)cc_malloc_shared_memory(&context, 100);
        std::cout<<"call tee_model_get_inputs"<<std::endl;
        auto ret = tee_model_get_inputs(&context, retval, data_buf, key_size, shape_vec);
        std::cout<<"finish tee_model_get_inputs"<<std::endl;
        if(ret!=0) {
          std::cerr<<"tee_model_get_inputs failed!"<<std::endl;
        }else{
          inputs = buildTensorFromBuff(shape_vec, data_buf);
        }
        ret = cc_free_shared_memory(&context, shape_vec);
        ret = cc_free_shared_memory(&context, data_buf);
        if(ret!=0){
          std::cerr << "cc_free_shared_memory failed." << std::endl;
        }
      } else { // ree get inputs
        inputs = this->ree_model_map[this->model_paths[i]]->GetInputs();
      }
      return inputs;
    }

    std::vector<mindspore::MSTensor> GetOutputsOf(int i, cc_enclave_t context, int *retval ){
      // return this->ree_model_map[this->model_paths[num_models-1]]->GetOutputs();
      std::vector<mindspore::MSTensor> outputs;
      if(this->backend_lists[i]=="TEE_CPU"){
        
        size_t key_size = this->model_paths[i].length();
        char *data_buf = (char *)cc_malloc_shared_memory(&context, 10000000);
        memcpy(data_buf,this->model_paths[i].c_str(),key_size);
        // set shape vec
        int *shape_vec = (int *)cc_malloc_shared_memory(&context, 100);
        // shape_vec: [num_tensors, num_dim_of_tensor1, tensor1.shape1, tensor1.shape2,.., num_dim_of_tensor_n,...]
        auto ret = tee_model_get_outputs(&context, retval, data_buf, key_size, shape_vec); 
        if(ret!=0) {
          std::cerr<<"tee_model_get_inputs failed!"<<std::endl;
        }else{
          outputs = buildTensorFromBuff(shape_vec, data_buf);
        }
        ret = cc_free_shared_memory(&context, shape_vec);
        ret = cc_free_shared_memory(&context, data_buf);
        if(ret!=0){
          std::cerr << "cc_free_shared_memory failed." << std::endl;
        }
      } else { // tee get outputs
        outputs = this->ree_model_map[this->model_paths[i]]->GetOutputs();
        
      }
      return outputs;
    }

    std::vector<mindspore::MSTensor> GetInputs(cc_enclave_t context, int *retval ){
      return this->GetInputsOf(0, context, retval);
    }

    std::vector<mindspore::MSTensor> GetOutputs(cc_enclave_t context, int *retval ){
      return this->GetOutputsOf(num_models-1, context, retval);
    }

    int Predict(const std::vector<mindspore::MSTensor> &inputs, std::vector<mindspore::MSTensor> *outputs, cc_enclave_t context, int *retval) {
      std::vector<mindspore::MSTensor> tmp_inputs;
      tmp_inputs = inputs;
      for(size_t i=0;i<num_models;i++) {
        if(backend_lists[i]=="TEE_CPU") {
          auto pred_ret = predictInCPUTEE(this->model_paths[i], tmp_inputs, outputs, context, retval);
          if(pred_ret!=0){
            std::cerr<<"predictInCPUTEE ERROR!"<<std::endl;
            return -1;
          }
        } else if (backend_lists[i]=="REE_CPU" || backend_lists[i]=="REE_CPU_D") {
          auto pred_ret = predictInCPUREE(this->ree_model_map[model_paths[i]],tmp_inputs, outputs);
          if(pred_ret!=0){
            std::cerr<<"predict ERROR!"<<std::endl;
            return -1;
          }
        } else if (backend_lists[i]=="REE_NPU") {
          auto pred_ret = predictInNPUREE(i, this->ree_model_map[model_paths[i]],tmp_inputs, outputs);
          if(pred_ret!=0){
            std::cerr<<"predictInNPUREE ERROR!"<<std::endl;
            return -1;
          }
        }
        tmp_inputs.assign(outputs->begin(),outputs->end());
      }
      return 0;
    }

    // int Predict(char* inputs_buf, size_t inputs_len, std::vector<mindspore::MSTensor> *outputs, size_t session_id, bool isPlaintxt, cc_enclave_t context, int *retval) {
    //   size_t output_len=0;
    //   char *outputs_buf = (char *)cc_malloc_shared_memory(&context, 10000000);
    //   for(size_t i=0;i<num_models;i++) {
    //     if(backend_lists[i]=="TEE_CPU") {
    //       if(i==0 && !isPlaintxt) {
    //         isPlaintxt = false;
    //       } else {
    //         isPlaintxt = true;
    //       }
    //       auto pred_ret = predictInCPUTEE(this->model_paths[i], inputs_buf, inputs_len, session_id, isPlaintxt, context, retval);
    //       if(pred_ret!=0){
    //         std::cerr<<"predictInCPUTEE ERROR!"<<std::endl;
    //         return -1;
    //       }
    //     } else if (backend_lists[i]=="REE_CPU" || backend_lists[i]=="REE_CPU_D") {
    //       auto pred_ret = predictInCPUREE(this->ree_model_map[model_paths[i]],inputs_buf, &output_len);
    //       if(pred_ret!=0){
    //         std::cerr<<"predict ERROR!"<<std::endl;
    //         return -1;
    //       }
    //       // memcpy(inputs_buf, outputs_buf, output_len);
    //     } else if (backend_lists[i]=="REE_NPU") {
    //       auto pred_ret = predictInNPUREE(i, this->ree_model_map[model_paths[i]],inputs_buf, context, retval);
    //       if(pred_ret!=0){
    //         std::cerr<<"predictInNPUREE ERROR!"<<std::endl;
    //         return -1;
    //       }
    //       // memcpy(inputs_buf, outputs_buf, output_len);
    //     }
        
    //   }
    //   *outputs = buildTensorFromBuff(*outputs, inputs_buf);
    //   auto ret = cc_free_shared_memory(&context, outputs_buf);
    //   if(ret!=0){
    //     std::cerr << "cc_free_shared_memory failed." << std::endl;
    //   }
    //   return 0;
    // }

    int Predict(char* inputs_buf, size_t inputs_len, std::vector<mindspore::MSTensor> *outputs, size_t session_id, bool inputPlaintxt, bool outputPlaintxt, cc_enclave_t context, int *retval) {
      for(size_t i=0;i<num_models;i++) {
        if(backend_lists[i]=="TEE_CPU") {
          if(i==0 && !inputPlaintxt) {
            inputPlaintxt = false;
          } else {
            inputPlaintxt = true;
          }
          if (i==num_models-1) {
            auto pred_ret = predictInCPUTEE(this->model_paths[i], inputs_buf, inputs_len, session_id, inputPlaintxt, outputPlaintxt, context, retval);
            if(pred_ret!=0){
              std::cerr<<"predictInCPUTEE ERROR!"<<std::endl;
              return -1;
            }
          } else {
            auto pred_ret = predictInCPUTEE(this->model_paths[i], inputs_buf, inputs_len, session_id, inputPlaintxt, true, context, retval);
            if(pred_ret!=0){
              std::cerr<<"predictInCPUTEE ERROR!"<<std::endl;
              return -1;
            }
          }
          
        } else if (backend_lists[i]=="REE_CPU") {
          auto pred_ret = predictInCPUREE(this->ree_model_map[model_paths[i]],inputs_buf);
          if(pred_ret!=0){
            std::cerr<<"predict ERROR!"<<std::endl;
            return -1;
          }
        } else if (backend_lists[i]=="REE_NPU") {
          auto pred_ret = predictInNPUREE(i, this->ree_model_map[model_paths[i]],inputs_buf, context, retval);
          if(pred_ret!=0){
            std::cerr<<"predictInNPUREE ERROR!"<<std::endl;
            return -1;
          }
        }
      }
      if (outputPlaintxt) {
        *outputs = buildTensorFromBuff(*outputs, inputs_buf);
      }
      
      return 0;
    }

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

    int predictInCPUTEE(std::string model_key, char* data_buf, size_t inputs_len, size_t session_id, bool inputPlaintxt, bool outputPlaintxt, cc_enclave_t context, int *retval) {
      std::cout<<"start predict In CPU TEE"<<std::endl;
      // create model key_buf
      size_t key_size = model_key.length();
      memcpy(data_buf + inputs_len, model_key.c_str(), key_size);
      // call tee to predict
      auto ret = tee_model_dec_predict_enc( &context, retval, key_size, data_buf, inputs_len, inputPlaintxt, outputPlaintxt, session_id);
      if(ret != CC_SUCCESS) {
        std::cerr << "tee_model_predict failed." << std::endl;
      }
      return 0;
    }

    int predictInCPUTEE(std::string model_key, char* data_buf, size_t inputs_len, size_t session_id, bool isPlaintxt , cc_enclave_t context, int *retval) {
      std::cout<<"start predict In CPU TEE"<<std::endl;
      // create model key_buf
      size_t key_size = model_key.length();
      char *key_buf = (char *)cc_malloc_shared_memory(&context, key_size);
      memcpy(key_buf,model_key.c_str(), key_size);
      // set shape vec
      int *shape_vec = (int *)cc_malloc_shared_memory(&context, 100);
      // call tee to predict
      auto ret = tee_model_dec_predict( &context, retval, key_buf, key_size, data_buf, inputs_len, isPlaintxt, session_id, shape_vec);
      if(ret != CC_SUCCESS) {
        std::cerr << "tee_model_predict failed." << std::endl;
      }
      ret = cc_free_shared_memory(&context, shape_vec);
      ret = cc_free_shared_memory(&context, key_buf);
      if(ret!=0){
        std::cerr << "cc_free_shared_memory failed." << std::endl;
      }
      return 0;
    }

    int predictInCPUTEE(std::string model_key, const std::vector<mindspore::MSTensor> &inputs, std::vector<mindspore::MSTensor> *outputs, cc_enclave_t context, int *retval) {
      std::cout<<"start predict In CPU TEE"<<std::endl;
      // create model key_buf
      size_t key_size = model_key.length();
      char *key_buf = (char *)cc_malloc_shared_memory(&context, key_size);
      memcpy(key_buf,model_key.c_str(), key_size);
      // create inputs data buf
      char *data_buf = (char *)cc_malloc_shared_memory(&context, 10000000);
      size_t data_len = 0;
      buildBufFromTensors(inputs, data_buf, &data_len);
      // set shape vec
      int *shape_vec = (int *)cc_malloc_shared_memory(&context, 100);
      // call tee to predict
      auto ret = tee_model_predict( &context, retval, key_buf, key_size, data_buf, shape_vec);
      if(ret != CC_SUCCESS) {
        std::cerr << "tee_model_predict failed." << std::endl;
      }
      // reconstruct output tensor from buffer
      auto tmp_outputs = buildTensorFromBuff(shape_vec, data_buf);
      outputs->assign(tmp_outputs.begin(),tmp_outputs.end());
      ret = cc_free_shared_memory(&context, shape_vec);
      ret = cc_free_shared_memory(&context, data_buf);
      ret = cc_free_shared_memory(&context, key_buf);
      if(ret!=0){
        std::cerr << "cc_free_shared_memory failed." << std::endl;
      }
      return 0;
    }

    int predictInCPUREE(mindspore::Model* model, const std::vector<mindspore::MSTensor> &inputs, std::vector<mindspore::MSTensor> *outputs) {
      std::cout<<"start predict In CPU REE"<<std::endl;
      auto pred_ret = model->Predict(inputs,outputs);
      if(pred_ret!=mindspore::kSuccess){
        std::cerr<<"predictInCPUREE ERROR!"<<std::endl;
        return -1;
      }
      return 0;
    }

    
    int predictInCPUREE(mindspore::Model* model, char* data_buf) {
      std::cout<<"start predict In CPU REE"<<std::endl;
      auto inputs = model->GetInputs();
      auto outputs = model->GetOutputs();
      inputs = buildTensorFromBuff(inputs, data_buf);
      auto pred_ret = model->Predict(inputs,&outputs);
      buildBufFromTensors(outputs, data_buf);
      if(pred_ret!=mindspore::kSuccess){
        std::cerr<<"predictInCPUREE ERROR!"<<std::endl;
        return -1;
      }
      return 0;
    }


    int predictInNPUREE(int model_idx, mindspore::Model* model, const std::vector<mindspore::MSTensor> &inputs, std::vector<mindspore::MSTensor> *outputs) {
      std::cout<<"start predict In NPU REE"<<std::endl;
      std::vector<mindspore::MSTensor> npu_inputs = inputs;
      if(model_idx>0) {
        if(backend_lists[model_idx-1] == "TEE_CPU") {
          npu_inputs = NCHW2NHWC(npu_inputs);
        }
      }
      auto pred_ret = model->Predict(npu_inputs,outputs);
      if(pred_ret!=mindspore::kSuccess){
        std::cerr<<"predictInCPUREE ERROR!"<<std::endl;
        return -1;
      }
      if(model_idx<(int)num_models-1){
        if(backend_lists[model_idx+1] == "TEE_CPU") {
          *outputs = NHWC2NCHW(*outputs);
        }
      }
      return 0;
    }

    int predictInNPUREE(int model_idx, mindspore::Model* model, char* data_buf,  cc_enclave_t context, int *retval) {
      std::cout<<"start predict In NPU REE"<<std::endl;
      // recover input tensor from buffer
      std::vector<mindspore::MSTensor> npu_inputs;
      if(model_idx>0) {
        if(backend_lists[model_idx-1] == "TEE_CPU") {
          auto last_outputs = GetOutputsOf(model_idx-1, context, retval);
          last_outputs = buildTensorFromBuff(last_outputs, data_buf);
          npu_inputs = NCHW2NHWC(last_outputs);
        }
      }else{
        npu_inputs = buildTensorFromBuff(npu_inputs, data_buf);
      }
      auto outputs = model->GetOutputs();
      auto pred_ret = model->Predict(npu_inputs, &outputs);
      if(pred_ret!=mindspore::kSuccess){
        std::cerr<<"predictInCPUREE ERROR!"<<std::endl;
        return -1;
      }
      if(model_idx<(int)num_models-1){
        if(backend_lists[model_idx+1] == "TEE_CPU") {
          outputs = NHWC2NCHW(outputs);
        }
      }
      size_t data_len = 0;
      buildBufFromTensors(outputs, data_buf, &data_len);
      return 0;
    }

};


int main(int argc, char** argv)
{
  int  retval = 0;
  char *path = (char*)PATH;
  std::cout<<"path = "<<path<<std::endl;
  //char buf[BUF_LEN];
  cc_enclave_t context = {SGX_ENCLAVE_TYPE_0};
    
  cc_enclave_result_t res = CC_FAIL;
  

  char real_p[PATH_MAX];
  /* check file exists, if not exist then use absolute path */
  if (realpath(path, real_p) == NULL) {
      if (getcwd(real_p, sizeof(real_p)) == NULL) {
          printf("Cannot find enclave.sign.so");
          return CC_FAIL;
      }
      if (PATH_MAX - strlen(real_p) <= strlen("/enclave.signed.so")) {
          printf("Failed to strcat enclave.sign.so path");
          return CC_FAIL;
      }
      (void)strcat(real_p, "/enclave.signed.so");
  }
  cc_sl_config_t sl_cfg = CC_USWITCHLESS_CONFIG_INITIALIZER;
  sl_cfg.num_tworkers = 2; /* 2 tworkers */
  sl_cfg.sl_call_pool_size_qwords = 2; /* 2 * 64 tasks */
  enclave_features_t features = {ENCLAVE_FEATURE_SWITCHLESS, (void *)&sl_cfg};

  std::cout<<"I am Here"<<std::endl;

  res = cc_enclave_create(real_p, AUTO_ENCLAVE_TYPE, 0, SECGEAR_DEBUG_FLAG, &features, 1, &context);

  std::cout<<"I am Here2"<<std::endl;

  // res = cc_enclave_create(real_p, AUTO_ENCLAVE_TYPE, 0, SECGEAR_DEBUG_FLAG, NULL, 0, &context);
  if (res != CC_SUCCESS) {
      printf("host create enclave error\n");
      return res;
      // goto end; 
  }

  // 指定输入模型路径
  // std::vector<std::string> model_paths{"/home/smliu/face-models/enc_models/yolo/part1_tee.ms", "/home/smliu/face-models/enc_models/yolo/part2_ree.ms"};
  // std::vector<std::string> backend_lists{"TEE_CPU", "REE_CPU"};
  // std::vector<std::string> model_paths{"/home/smliu/face-models/0822/subgraph_landmark.ms", "/home/smliu/face-models/0822/landmark_nw_0.ms"};
  // std::vector<std::string> backend_lists{"TEE_CPU", "TEE_CPU"};
  // std::vector<std::string> model_paths{"/home/smliu/face-models/0904/subgraph_sw_0.ms", "/home/smliu/face-models/0904/subgraph_nw_0.ms", "/home/smliu/face-models/0904/subgraph_sw_1.ms"};
  // std::vector<std::string> backend_lists{"TEE_CPU", "REE_CPU", "TEE_CPU"};
/*
  std::vector<std::string> model_paths{"/home/hanzhibin/interface-build-predict/test_models/subgraph_sw_0.ms",
                                       "/home/hanzhibin/interface-build-predict/test_models/subgraph_nw_0.ms",
                                       "/home/hanzhibin/interface-build-predict/test_models/subgraph_sw_1.ms"};
  std::vector<std::string> backend_lists{"REE_CPU", "REE_CPU", "REE_CPU"};
*/
  // std::vector<std::string> backend_lists{"REE_CPU_D", "REE_CPU", "REE_CPU_D"};



  // std::vector<std::string> model_paths{"/home/smliu/face-models/backbone_encrypt/subgraph_sw_0.ms"};
  // std::vector<std::string> backend_lists{"TEE_CPU"};
  
  // std::vector<std::string> model_paths{"/home/smliu/face-models/0822/subgraph_backbone_0.ms", "/home/smliu/face-models/0822/backbone_nw_0.ms", "/home/smliu/face-models/0822/subgraph_backbone_1.ms"};
  // std::vector<std::string> backend_lists{"TEE_CPU", "REE_CPU", "TEE_CPU"};

  // std::vector<std::string> model_paths{"/home/smliu/face-models/0822/subgraph_yolo.ms", "/home/smliu/face-models/0822/yolo_nw_0.ms"};
  // std::vector<std::string> backend_lists{"TEE_CPU", "REE_CPU"};

  std::vector<std::string> model_paths{"/home/hanzhibin/model_test/bert-ms/bert-base-chinese-onelayer.ms"};
  std::vector<std::string> backend_lists{"TEE_CPU"};

  CCModel landmark = CCModel(model_paths, backend_lists);

  auto build_ret = landmark.Build( context, &retval);
  if(build_ret!=0){
    std::cerr<<"CCModelBuild ERROR!"<<std::endl;
    return build_ret;
  }
  std::cout << "build success." << std::endl;

  auto inputs = landmark.GetInputs(context, &retval);
  std::cout << "get inputs success." << std::endl;

  auto outputs = landmark.GetOutputs(context, &retval);

  auto ret=GenerateInputDataWithRandom(inputs);
  if (ret != mindspore::kSuccess) {
      std::cerr << "Generate Random Input Data failed." << std::endl;
      return -1;
  }

  // landmark.Predict(inputs, &outputs, context, &retval);
  std::cout<< "inputs size="<<inputs[0].DataSize()<<std::endl;

// data in buff format not tensor
  char *inputs_buf = (char *)cc_malloc_shared_memory(&context, 10000000);
  size_t input_len = 0;
  landmark.buildBufFromTensors(inputs, inputs_buf, &input_len);
  landmark.Predict(inputs_buf, input_len, &outputs, 0, true, true, context, &retval);
  res = cc_free_shared_memory(&context, inputs_buf);


  for (auto tensor : outputs) {
    std::cout << "tensor name is:" << tensor.Name() << " tensor size is:" << tensor.DataSize()
            << " tensor elements num is:" << tensor.ElementNum() << std::endl;
    auto out_data = reinterpret_cast<const float *>(tensor.Data().get());
    std::cout << "output data is:";
    for (int i = 0; i < tensor.ElementNum() && i <= 200; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }

  res = cc_enclave_destroy(&context);
  if(res != CC_SUCCESS) {
      printf("host destroy enclave error\n");
  } else {
      printf("host destroy enclave success\n");
  }
  return res;
}




