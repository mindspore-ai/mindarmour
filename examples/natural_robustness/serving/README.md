# 自然扰动样本生成serving服务

提供自然扰动样本生成在线服务。客户端传入图片和扰动参数，服务端返回扰动后的图片数据。

## 环境准备

硬件环境：Ascend 910，GPU

操作系统：Linux-x86_64

软件环境：

1. python 3.7.5或python 3.9.0

2. 安装MindSpore 1.6.0可以参考[MindSpore安装页面](https://www.mindspore.cn/install)

3. 安装MindSpore Serving 1.6.0可以参考[MindSpore Serving 安装页面](https://www.mindspore.cn/serving/docs/zh-CN/r1.5/serving_install.html)

4. 安装serving分支的MindArmour:

   - 从Gitee下载源码

     `git clone https://gitee.com/mindspore/mindarmour.git`

   - 编译并安装MindArmour

     `python setup.py install`

### 文件结构说明

```bash
serving
├── server
│   ├── serving_server.py   # 启动serving服务脚本
│   └── perturbation
│       └── serverable_config.py    # 服务端接收客户端数据后的处理脚本
└── client
    ├── serving_client.py   # 启动客户端脚本
    └── perturb_config.py   # 扰动方法配置文件
```

## 脚本说明及使用

### 部署Serving推理服务

1. #### `servable_config.py`说明。

   ```python
   ···

   # 客户端可以请求的方法，包含4个返回值："results", "file_names", "file_length", "names_dict"

   @register.register_method(output_names=["results", "file_names", "file_length", "names_dict"])
   def natural_perturbation(img, perturb_config, methods_number, outputs_number):
       """method natural_perturbation data flow definition, only preprocessing and call model"""
       res = register.add_stage(perturb, img, perturb_config, methods_number, outputs_number, outputs_count=4)
       return res
   ```

   方法`natural_perturbation`为对外提供服务的接口。

   **输入：**

   - img：输入为图片，格式为bytes。
   - perturb_config：扰动配置项，具体配置参考`perturb_config.py`。
   - methods_number：每次扰动随机从配置项中选择方法的个数。
   - outputs_number：对于每张图片，生成的扰动图片数量。

   **输出**res中包含4个参数：

   - results：拼接后的图像bytes；

   - file_names：图像名，格式为`xxx.png`，其中‘xxx’为A-Za-z中随机选择20个字符构成的字符串。

   - file_length：每张图片的bytes长度。

   - names_dict: 图片名和图片使用扰动方法构成的字典。格式为：

     ```bash
     {
     picture1.png: [[method1, parameters of method1], [method2, parameters of method2], ...]],
     picture2.png: [[method3, parameters of method3], [method4, parameters of method4], ...]],
     ...
     }
     ```

2. #### 启动server。

   ```python
   ···

   def start(address):
       servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
       # 服务配置
       servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="perturbation", device_ids=(0, 1), num_parallel_workers=4)
       # 启动服务
       server.start_servables(servable_configs=servable_config)

       # 启动启动gRPC服务，用于客户端和服务端之间通信
       server.start_grpc_server(address=address, max_msg_mb_size=200)   # ip和最大的传输数据量，单位MB
       # 启动启动Restful服务，用于客户端和服务端之间通信
       server.start_restful_server(address=address)
   ```

   gRPC传输性能更好，Restful更适合用于web服务，根据需要选择。

   执行命令`python serverong_server.py`启动服务。

   当服务端打印日志`Serving RESTful server start success, listening on *.*.*.*:****`时，表示Serving RESTful服务启动成功，推理模型已成功加载。

### 客户端进行推理

1. 在`perturb_config.py`中设置扰动方法及参数。下面是个例子：

   ```python
   PerturbConfig = [{"method": "Contrast", "params": {"alpha": 1.5, "beta": 0}},
                    {"method": "GaussianBlur", "params": {"ksize": 5}},
                    {"method": "SaltAndPepperNoise", "params": {"factor": 0.05}},
                    {"method": "Translate", "params": {"x_bias": 0.1, "y_bias": -0.2}},
                    {"method": "Scale", "params": {"factor_x": 0.7, "factor_y": 0.7}},
                    {"method": "Shear", "params": {"factor": 2, "director": "horizontal"}},
                    {"method": "Rotate", "params": {"angle": 40}},
                    {"method": "MotionBlur", "params": {"degree": 5, "angle": 45}},
                    {"method": "GradientBlur", "params": {"point": [50, 100], "kernel_num": 3, "center": True}},
                    {"method": "GradientLuminance",
                     "params": {"color_start": [255, 255, 255],
                                "color_end": [0, 0, 0],
                                "start_point": [100, 150], "scope": 0.3,
                                "bright_rate": 0.3, "pattern": "light",
                                "mode": "circle"}},
                    {"method": "Curve", "params": {"curves": 5, "depth": 10,
                                                    "mode": "vertical"}},
                    {"method": "Perspective",
                     "params": {"ori_pos": [[0, 0], [0, 800], [800, 0], [800, 800]],
                                "dst_pos": [[50, 0], [0, 800], [780, 0], [800, 800]]}},
                   ]
   ```

   其中`method`为扰动方法名，`params`为对应方法的参数。可用的扰动方法及对应参数可在`mindarmour/natural_robustness/natural_noise.py`中查询。

2. 在`serving_client.py`中写客户端的处理脚本，包含输入输出的处理、服务端的调用，可以参考下面的例子。

   ```python
   ···

   def perturb(perturb_config, address):
       """invoke servable perturbation method natural_perturbation"""

       # 请求的服务端ip及端口、请求的服务名、请求的方法名
       client = Client(address, "perturbation", "natural_perturbation")

       # 输入数据
       instances = []
       img_path = '/root/mindarmour/example/adversarial/test_data/1.png'
       result_path = '/root/mindarmour/example/adv/result/'
       methods_number = 2
       outputs_number = 3
       img = cv2.imread(img_path)
       img = cv2.imencode('.png', img)[1].tobytes()    # 图片传输用bytes格式，不支持numpy.ndarray格式
       perturb_config = json.dumps(perturb_config) # 配置方法转成json格式
       instances.append({"img": img, 'perturb_config': perturb_config, "methods_number": methods_number,
                      "outputs_number": outputs_number})   # instances中可添加多个输入

       # 请求服务，返回结果
       result = client.infer(instances)

       # 对服务请求得到的结果进行处理，将返回的图片字节流存成图片
       file_names = result[0]['file_names'].split(';')
       length = result[0]['file_length'].tolist()
       before = 0
       for name, leng in zip(file_names, length):
           res_img = result[0]['results']
           res_img = res_img[before:before + leng]
           before = before + leng
           print('name: ', name)
           image = Image.open(BytesIO(res_img))
           image.save(os.path.join(result_path, name))

       names_dict = result[0]['names_dict']
       with open('names_dict.json', 'w') as file:
           file.write(names_dict)
   ```

   启动client前，需将服务端的IP地址改成部署server的IP地址，图片路径、结果存储路基替换成用户数据路径。

   目前serving数据传输支持的数据类型包括：python的int、float、bool、str、bytes，numpy number, numpy array object。

   输入命令`python serving_client.py`开启客户端，如果对应目录下生成扰动样本图片则说明serving服务正确执行。

   ### 其他

   在`serving_logs`目录下可以查看运行日志，辅助debug。
