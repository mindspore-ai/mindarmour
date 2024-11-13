"""launch gradient inversion attacks with different configs."""
import breaching
import mindspore as ms
from custom_dataset import CustomData


def reset_cfg_by_client(cfg, args):
    """reset default configs by input parameters."""
    cfg.case.user.num_data_points = int(args['custom_parameter']['num_data_points'])
    cfg.attack.optim.max_iterations = int(args['custom_parameter']['max_iterations'])
    cfg.attack.optim.step_size = float(args['custom_parameter']['step_size'])
    cfg.attack.regularization.total_variation.scale = float(args['custom_parameter']['TV_scale'])
    cfg.attack.regularization.total_variation.tv_start = int(args['custom_parameter']['TV_start'])
    cfg.attack.optim.callback = int(args['custom_parameter']['callback'])
    if 'invgrad' not in args['alg_name'].lower():
        cfg.attack.regularization.deep_inversion.scale = float(args['custom_parameter']['BN_scale'])
        cfg.attack.regularization.deep_inversion.deep_inv_start = int(args['custom_parameter']['BN_start'])


def get_cfg(args):
    """get configs."""
    if 'cifar' in args['dataset'].lower():
        case_ = "case=11_small_batch_cifar"
    elif 'tiny' in args['dataset'].lower():
        case_ = "case=13_custom_tinyimagenet"
    elif 'web' in args['dataset'].lower():
        case_ = "case=12_custom_imagenet"
    else:
        raise ValueError(f"Do not support dataset: {args['dataset']}!")
    if 'invgrad' in args['alg_name'].lower():
        attack_ = "attack=invertinggradients"
    elif 'seethrough' in args['alg_name'].lower():
        attack_ = "attack=seethrough_res18" if "resnet18" in args['model'].lower() else "attack=seethrough_res34"
    elif 'stepwise' in args['alg_name'].lower():
        attack_ = "attack=stepwise_res18" if "resnet18" in args['model'].lower() else "attack=stepwise_res34"
    else:
        raise ValueError(f"Do not support algorithm: {args['alg_name']}!")
    return breaching.get_config(overrides=[case_, attack_])


def run_attack(args):
    """run attacks based on the configs."""
    try:
        ms.set_context(device_target="GPU", device_id=0)
        print('using mindspore with GPU context')
    except ValueError:
        ms.set_context(device_target="CPU")
        print('using mindspore with CPU context')

    cfg = get_cfg(args)
    data_dir = './custom_data/1_img/'

    cfg.attack.save_dir = args['out_put']
    cfg.case.data.path = args['data_path']

    cfg.case.data.partition = 'balanced'
    cfg.case.data.smooth = 0
    cfg.case.user.user_idx = 0
    cfg.case.model = args['model']

    cfg.case.user.provide_labels = False
    cfg.case.user.provide_buffers = False
    cfg.case.server.provide_public_buffers = True
    cfg.case.server.pretrained = False

    # ---------根据用户自定义重置参数---------------------
    reset_cfg_by_client(cfg, args)
    cus_defense = args['custom_parameter']['defense']
    sercure_input, apply_noise, apply_prune = False, False, False
    if cus_defense == 'Vicinal Augment':
        sercure_input = True
    if cus_defense == 'Differential Privacy':
        apply_noise = True
    if cus_defense == 'Gradient Prune':
        apply_prune = True
    # --------------run--------------------------------
    user, server = breaching.cases.construct_case(cfg.case)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack)
    attacker.txt_path = args['out_put'] + args['eva_txt_file']
    server_payload = server.distribute_payload()

    if 'web' not in args['dataset'].lower():
        cus_data = CustomData(data_dir=data_dir, dataset_name=args['dataset'],
                              number_data_points=cfg.case.user.num_data_points)
        shared_data, true_user_data = user.compute_local_updates(server_payload, secure_input=sercure_input,
                                                                 apply_noise=apply_noise, apply_prune=apply_prune)
    else:
        cus_data = CustomData(data_dir=data_dir, dataset_name='ImageNet',
                              number_data_points=cfg.case.user.num_data_points)
        shared_data, true_user_data = user.compute_local_updates(server_payload, custom_data=cus_data.process_data(),
                                                                 secure_input=sercure_input,
                                                                 apply_noise=apply_noise, apply_prune=apply_prune)

    true_pat = cfg.attack.save_dir + 'A_0.png'
    cus_data.save_recover(true_user_data, save_pth=true_pat)

    ## ---------------attack part---------------------
    reconstructed_user_data = attacker.reconstruct([server_payload], [shared_data], {}, custom=cus_data)
    recon_path__ = cfg.attack.save_dir + f'A_{attacker.save_flag}.png'
    cus_data.save_recover(reconstructed_user_data, true_user_data, recon_path__)
