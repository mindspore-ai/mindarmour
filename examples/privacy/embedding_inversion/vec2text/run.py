'''
program start point
'''

from datetime import datetime
from experiments import experiment_from_args
from run_args import parse_arguments, parse_args_into_dataclasses

# #为了在gpu环境下使用mindspore2.4版本
# os.environ['CUDA_HOME'] = '/luoyf'
#
# os.environ["WANDB_DISABLED"] = "true"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# #连接clash挂梯子必备
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'


def main():
    # ms.set_context(device_target="Ascend")
    # ms.context.set_context(mode=ms.context.PYNATIVE_MODE)
    # ms.context.set_context(mode=ms.context.GRAPH_MODE)
    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", from_tf=True)
    args = parse_arguments()
    #traing_args may not be a normal datacla
    # ss, and then should be adapted to the new one.
    model_args, data_args, training_args = parse_args_into_dataclasses(args)
    experiment = experiment_from_args(model_args, data_args, training_args)
    print("beginning time:")
    print(datetime.now())
    experiment.run()



if __name__ == "__main__":
    main()
