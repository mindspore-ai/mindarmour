# Dataset

Dataset used: [COCO2017](<https://cocodataset.org/>) 

- Dataset size：19G
  - Train：18G，118000 images  
  - Val：1G，5000 images 
  - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
  - Note：Data will be processed in dataset.py

# Environment Requirements

- Install [MindSpore](https://www.mindspore.cn/install/en).

- Download the dataset COCO2017.

- We use COCO2017 as dataset in this example.

    Install Cython and pycocotool, and you can also install mmcv to process data.

    ```
    pip install Cython

    pip install pycocotools

    pip install mmcv==0.2.14
    ```

    And change the COCO_ROOT and other settings you need in `config.py`. The directory structure is as follows:

    ```
    .
    └─cocodataset
      ├─annotations
        ├─instance_train2017.json
        └─instance_val2017.json
      ├─val2017
      └─train2017    
    ```

# Quick start
You can download the pre-trained model checkpoint file [here](<https://www.mindspore.cn/resources/hub/details?2505/MindSpore/ascend/0.7/fasterrcnn_v1.0_coco2017>).
```
python coco_attack_pgd.py --pre_trained [PRETRAINED_CHECKPOINT_FILE] 
```
> Adversarial samples will be generated.
