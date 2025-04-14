from  mindformers.models.llama.llama_transformer import LLamaDecodeLayer
from  mindformers.models.llama.llama_layer import LlamaEmbedding, LlamaRMSNorm, FreqsMgr
from  mindformers.modules.transformer.transformer import LowerTriangularMaskWithDynamic
from mindspore import nn, __version__, ops
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import LossMonitor
import mindspore.dataset as ds
import mindspore
from mindspore import Tensor
import numpy as np
from tqdm import tqdm    
from mindspore import save_checkpoint
import argparse

class LLamalayer0(nn.Cell):
    def __init__(self):
        super(LLamalayer0, self).__init__()  
        self.layer = LLamaDecodeLayer(  1,
                                        4096,
                                        0,
                                        dim=4096,
                                        n_heads=1,
                                        multiple_of=256,
                                        norm_eps=1.0e-5,
                                        qkv_has_bias=False,
                                        qkv_concat=False,
                                        compute_dtype=mindspore.float16,
                                        layernorm_compute_dtype=mindspore.float32,
                                        softmax_compute_dtype=mindspore.float16,
                                        rotary_dtype=mindspore.float16,
                                        param_init_type=mindspore.float16,
                                        use_past=False,
                                        use_flash_attention=False,
                                        use_paged_attention=False,
                                        block_size=16,
                                        num_blocks=512,
                                        is_dynamic=False,
                                        use_kvcache_op=False,
                                        is_flexible_shape=False,
                                        use_rope_slice=False,)
        self.freqs_mgr = FreqsMgr(head_dim=4096,
                                  seq_length=4096,
                                  max_position_embedding=4096,
                                  rotary_dtype=mindspore.float16,
                                  theta=10000.0,
                                  scaling_factor=1.0,
                                  extend_method='None',
                                  is_dynamic=False)
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=4096,
                                                          compute_type=mindspore.float16,
                                                          is_dynamic=False,
                                                          pad_token_id=0,
                                                          use_flash_attention=False)
        
    def construct(self, x, x2):
        print("x.shape:",x.shape) #(1, 1, 4096, 4096)
        print("x2.shape",x2.shape) 
        freqs_cis = self.freqs_mgr()
        print("freqs_cis",freqs_cis)
        mask = self.casual_mask(x2) # mask: [bs, seq, seq]
        mask = self.casual_mask.post_process(mask)
        print("mask",mask)
        kvcache_inputs = None
        output = self.layer(x, freqs_cis, mask, kvcache_inputs)   
        return output
        

def generator():
    for i in range(len(input1)):
        yield (
            Tensor(input1[i], mindspore.float16),      
            Tensor(input2[i][np.newaxis, :], mindspore.float16),  
            Tensor(output[i], mindspore.float16)              
        )
        
class TrainNetWithLoss(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(TrainNetWithLoss, self).__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, input1, input2, label):
        out = self.backbone(input1, input2)
        return self.loss_fn(out, label)
        
class MyDataset:
    def __init__(self, input1, input2, output):
        self.input1 = input1
        self.input2 = input2
        self.output = output

    def __getitem__(self, index):
        input1_clean = np.squeeze(self.input1[index], axis=0)  # (4096, 4096)
        input2_clean = np.squeeze(self.input2[index], axis=0)  # (4096,)
        output_clean = np.squeeze(self.output[index], axis=0)  # (4096, 4096)

        return (Tensor(input1_clean, mindspore.float16),
                Tensor(input2_clean, mindspore.float16),
                Tensor(output_clean, mindspore.float16))

    def __len__(self):
        return len(self.input1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz",type=str,help="SEM Track dateset",)
    args = parser.parse_args()

    dataset_npz = np.load(args.dataset_npz)
    input1 = dataset_npz["input1"]  
    input2 = dataset_npz["input2"]  
    output = dataset_npz["output"]

    my_dataset = MyDataset(input1, input2, output)
    train_dataset = ds.GeneratorDataset(my_dataset, ["input1", "input2", "label"])
    train_dataset = train_dataset.batch(1)

    from mindspore import nn
    loss_fn = nn.MSELoss()
    model = LLamalayer0()
    net_with_loss = TrainNetWithLoss(model, loss_fn)
    optimizer = nn.Adam(net_with_loss.trainable_params(), learning_rate=1e-4)
    
    train_net = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_net.set_train()

    for epoch in range(5):  
        print(f"\n🔁 Epoch {epoch+1}")
        num_batches = train_dataset.get_dataset_size()
        progress_bar = tqdm(train_dataset.create_dict_iterator(), total=num_batches, desc=f"Epoch {epoch+1}")
   
        for data in progress_bar:
            input1 = data["input1"]
            input2 = data["input2"]
            label = data["label"]
            loss = train_net(input1, input2, label)
            progress_bar.set_postfix(loss=loss.asnumpy().item())
        save_checkpoint(model, f"sem_track_epoch_{epoch + 1}.ckpt")
           
  