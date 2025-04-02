### CmakeList配置
1. 在`secgear/CMakeLists.txt`中配置正确的GP_SDK_DEFAULT_PATH路径：
```
set(SGX_SDK_DEFAULT_PATH /opt/intel/sgxsdk)
set(GP_SDK_DEFAULT_PATH ${...}/sdk/itrustee_sdk)
set(PL_SDK_DEFAULT_PATH /root/dev/sdk)
```
2. 在`examples/helloworld/CMakeLists.txt`中增加C++编译配置
```
project(HelloWorld CXX C)
...
set(CMAKE_CXX_STANDARD 17)
```
3. 在`examples/helloworld/host/CMakeLists.txt`中配置正确的三方库路径和编译选项
4. 在`examples/helloworld/enclave/CMakeLists.txt`中配置正确的三方库路径和编译选项
5. 修改证书配置