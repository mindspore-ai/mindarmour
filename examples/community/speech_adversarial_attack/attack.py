"""
Speech attack
"""
import json
import datetime
import numpy as np
from g2p_en import G2p
from tqdm import tqdm

import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from mindarmour.utils.logger import LogUtil

import stft
from src.deepspeech2 import DeepSpeechModel, PredictWithSoftmax
from src.greedydecoder import MSGreedyDecoder
from src.config import eval_config
from dataset import create_dataset


context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

LOGGER = LogUtil.get_instance()
TAG = "SpeechAttack"

SAMLE_RATE = 16000
WINDOW_SIZE = 0.02
WINDOW_STRIDE = 0.01
WINDOW = "hamming"
N_FFT = int(SAMLE_RATE * WINDOW_SIZE)
WIN_LENGTH = N_FFT
HOP_LENGTH = int(SAMLE_RATE * WINDOW_STRIDE)
config = eval_config

stft_fn = stft.STFT(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window="hamming",
    center=True,
    pad_mode="reflect",
)

stft_fn.bprop_debug = True

with open("labels.json") as label_file:
    labels = json.load(label_file)

model = DeepSpeechModel(
    batch_size=config.DataConfig.batch_size,
    rnn_hidden_size=config.ModelConfig.hidden_size,
    nb_layers=config.ModelConfig.hidden_layers,
    labels=labels,
    rnn_type=config.ModelConfig.rnn_type,
    audio_conf=config.DataConfig.SpectConfig,
    bidirectional=True,
)
model_with_predict = PredictWithSoftmax(model)
decoder = MSGreedyDecoder(labels=labels, blank_index=labels.index("_"))
target_decoder = MSGreedyDecoder(labels, blank_index=labels.index("_"))

param_dict = load_checkpoint("./checkpoints/DeepSpeech2_model.ckpt")
param_dict_new = {}
for k, v in param_dict.items():
    if "rnn" in k:
        new_k = k.replace("rnn", "RNN")
        param_dict_new[new_k] = param_dict[k]
    else:
        param_dict_new[k] = param_dict[k]
load_param_into_net(model, param_dict_new)

LOGGER.info(TAG, "Successfully loading the pre-trained model")

model.set_train(False)
model.bprop_debug = True

concat_op = ms.ops.Concat(axis=1)


def superpose(audio, noise):
    """
    Add noise into the audio.

    Args:
        audio : audio.
        noise : noise.

    Returns:
        res: The attack audio.
        start: The start position.
    """
    audio_len = audio.shape[1]
    noise_len = noise.shape[1]
    start = np.random.randint(0, noise_len)
    while noise_len < audio_len:
        noise = np.concatenate((noise, noise), axis=1)
        noise_len = noise.shape[1]
    noise = np.concatenate((noise, noise), axis=1)
    noise = noise[:, start : audio_len + start]

    res = audio + noise
    res = np.clip(res, -1, 1)
    return res, start


def sr_and_cer(sound_data, noise):
    """
    Calculate the attacking success rate (SR) and the character error rate (CER).
    """
    i = 0
    cer = 0.0
    sound_len = len(sound_data)
    for sound in tqdm(sound_data):
        label = sound["label"]
        if noise is not None:
            new_sound = superpose(np.array(sound["wave"]), noise)[0]
        else:
            new_sound = np.array(sound["wave"])

        new_sound = stft.ms_spectrogram(ms.Tensor(new_sound), stft_fn)

        input_size = new_sound.shape[-1]
        input_size = ms.Tensor([input_size])

        out, out_size = model_with_predict(new_sound, input_size)
        decoded_output, _ = decoder.decode(out, out_size)

        pred_cer = decoder.cer(decoded_output[0][0], label) / len(label)
        cer += pred_cer
        if pred_cer > 0.5:
            i += 1
    LOGGER.info(TAG, "SR = " + str(i / sound_len))
    LOGGER.info(TAG, "CER = " + str(cer / sound_len))


def generate_data(num, alpha):
    """
    Generate training dataset.
    """
    g2p = G2p()

    ds_eval = create_dataset(
        audio_conf=config.DataConfig.SpectConfig,
        manifest_filepath=config.DataConfig.test_manifest,
        labels=labels,
        normalize=True,
        train_mode=False,
        batch_size=1,
        rank=0,
        group_size=1,
    )

    LOGGER.info(TAG, str(ds_eval.get_dataset_size()))

    train_data = []
    index = 0
    s_num_sum = 0
    s_len_sum = 0
    for data in ds_eval.create_dict_iterator():
        sound, target_indices, targets = (
            data["inputs"],
            data["target_indices"],
            data["label_values"],
        )
        sound = sound.asnumpy()
        target_indices, targets = target_indices.asnumpy(), targets.asnumpy()

        label = target_decoder.convert_to_strings([list(targets)])

        phoneme_list = g2p(label)
        s_num_sum += len(phoneme_list)
        s_len_sum += sound.shape[1] / SAMLE_RATE

    density_mean = s_num_sum / s_len_sum

    for data in ds_eval.create_dict_iterator():
        sound, target_indices, targets = (
            data["inputs"],
            data["target_indices"],
            data["label_values"],
        )
        sound = sound.asnumpy()
        target_indices, targets = target_indices.asnumpy(), targets.asnumpy()

        label = target_decoder.convert_to_strings([list(targets)])

        phoneme_list = g2p(label)
        s_num = len(phoneme_list)
        s_len = sound.shape[1] / SAMLE_RATE
        density = s_num / s_len
        if density < density_mean - alpha or density > density_mean + alpha:
            continue
        if index < num:
            train_data.append(
                {
                    "wave": sound,
                    "target_indices": data["target_indices"],
                    "label_values": data["label_values"],
                }
            )
            LOGGER.info(TAG, str(label))
        else:
            break
        index += 1
    return train_data


class MyAdam(ms.nn.Cell):
    """
    Adam for Mindspore.
    """

    def __init__(self, param):
        super(MyAdam, self).__init__()
        self.apply_adam = ms.ops.Adam()
        self.param_m = ms.Parameter(
            ms.Tensor(np.zeros(param.shape).astype(np.float32)), name="param_m"
        )
        self.param_v = ms.Parameter(
            ms.Tensor(np.zeros(param.shape).astype(np.float32)), name="param_v"
        )

    def construct(
            self, var, beta1_power, beta2_power, learning_rate, beta1, beta2, epsilon, grad
    ):
        """
        Run.
        """
        val, self.param_m, self.param_v = self.apply_adam(
            var,
            self.param_m,
            self.param_v,
            beta1_power,
            beta2_power,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            grad,
        )
        return val


def superpose_step(sound, noise, start):
    """
    Add noise into the audio in the step.

    Args:
        audio : Audio.
        noise : Noise.
        start : Add noise in the step.

    Returns:
        res: The attack audio.
        start: The start position.
    """
    sound_len = sound.shape[1]
    noise_len = noise.shape[1]
    while noise_len < sound_len:
        noise = concat_op((noise, noise))
        noise_len = noise.shape[1]
    noise = concat_op((noise, noise))
    noise = noise[:, start : sound_len + start]

    res = sound + noise
    res = ms.ops.clip_by_value(res, -1, 1)
    return res, start


def generate(
        train_data,
        epoch_size,
        adv_len,
        eps,
        step,
        iteration=100,
        learning_rate=1e-1,
):
    """
    Generate speech attack noise.

    Args:
        train_data : Training data.
        epoch_size : Training epochs.
        adv_len : The attack noise length.
        eps : Attack eps.
        step : Attack steps.
        iteration : Iteration steps.
        learning_rate : Learning rate.

    Returns:
        res: The attack audio,
        start: The start position.
    """
    noise_init = train_data[0]["wave"][:, :adv_len]
    noise = ms.Tensor(noise_init)

    noise = ms.ops.clip_by_value(noise, -eps, eps)

    myadam = MyAdam(noise)

    data_length = len(train_data)

    loss_fn = P.CTCLoss(ctc_merge_repeated=True)
    reduce_mean_false = P.ReduceMean(keep_dims=False)
    cast_op = P.Cast()
    inference_softmax = P.Softmax(axis=-1)
    transpose_op = P.Transpose()

    for epoch in range(epoch_size):
        for index, sound in enumerate(train_data):
            one_sound = ms.Tensor(sound["wave"])

            target_indices, targets = sound["target_indices"], sound["label_values"]

            def forward_fn(
                    one_sound, noise, input_size, step_start, target_indices, label_values
            ):
                advs, _ = superpose_step(one_sound, noise, step_start)
                advs = stft.ms_spectrogram(advs, stft_fn)
                input_size[0] = advs.shape[-1]

                predict, output_length = model(advs, input_size)

                loss = loss_fn(
                    predict,
                    target_indices,
                    label_values,
                    cast_op(output_length, mstype.int32),
                )
                loss = -reduce_mean_false(loss[0])
                return loss, predict, output_length

            grad_fn = ms.value_and_grad(forward_fn, 1, None, has_aux=True)

            step_start = 0
            for cur_index in range(iteration):
                input_size = ms.Tensor([one_sound.shape[-1]])

                (loss, out, out_size), grads = grad_fn(
                    one_sound, noise, input_size, step_start, target_indices, targets
                )

                noise = myadam(
                    noise, 0.9, 0.999, learning_rate, 0.9, 0.999, 1e-8, grads
                )

                loss = ms.ops.depend(loss, noise)

                noise = noise.clip(-eps, eps)

                out = inference_softmax(out)
                out = transpose_op(out, (1, 0, 2))

                decoded_output, _ = decoder.decode(out, out_size)
                label = target_decoder.convert_to_strings([list(targets.asnumpy())])

                LOGGER.info(TAG, str(decoded_output))
                LOGGER.info(TAG, str(label))

                cer_inst = decoder.cer(decoded_output[0][0], label[0][0])
                cer = cer_inst / len(label[0][0].replace(" ", ""))
                LOGGER.info(TAG, str(cer))

                step_start += step
                if step_start > adv_len:
                    step_start -= adv_len

                LOGGER.info(
                    TAG,
                    "epoch = {}/{}; data = {}/{}; iter = {}/{}; loss = {};"
                    " cur_cer = {:.4f}; decode_out = {}".format(
                        epoch + 1,
                        epoch_size,
                        index + 1,
                        data_length,
                        cur_index + 1,
                        iteration,
                        loss,
                        cer,
                        decoded_output[0][0],
                    ),
                )
    return noise.asnumpy()


if __name__ == "__main__":
    test_dataset = np.load("source/100_test_audio_list.npy", allow_pickle=True)

    sr_and_cer(test_dataset, None)
    train_dataset = generate_data(10, 0.2)

    start_time = datetime.datetime.now()
    attack_noise = generate(
        train_dataset,
        epoch_size=3,
        adv_len=3200,
        eps=0.05,
        step=2467,
        iteration=30,
        learning_rate=2e-3,
    )
    end_time = datetime.datetime.now()
    time = end_time - start_time
    LOGGER.info(TAG, "Time = " + str(end_time - start_time))

    sr_and_cer(test_dataset, attack_noise)
