# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Create train or eval dataset.
"""
import math

import numpy as np
import mindspore.dataset.engine as de
import librosa
import soundfile as sf

TRAIN_INPUT_PAD_LENGTH = 1250
TRAIN_LABEL_PAD_LENGTH = 350
TEST_INPUT_PAD_LENGTH = 3500


class LoadAudioAndTranscript:
    """
    Parse audio and transcript.
    """

    def __init__(self, audio_conf=None, normalize=False, labels=None):
        super(LoadAudioAndTranscript, self).__init__()
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window
        self.is_normalization = normalize
        self.labels = labels

    @staticmethod
    def load_audio(path):
        """
        Load audio.
        """
        sound, _ = sf.read(path, dtype="int16")
        sound = sound.astype("float32") / 32767
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)
        return sound

    def parse_audio(self, audio_path):
        """
        Parse audio.
        """
        audio = self.load_audio(audio_path)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        d = librosa.stft(
            y=audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
        )
        mag, _ = librosa.magphase(d)
        mag = np.log1p(mag)
        if self.is_normalization:
            mean = mag.mean()
            std = mag.std()
            mag = (mag - mean) / std
        return mag

    def parse_transcript(self, transcript_path):
        """
        Parse_transcript.
        """
        with open(transcript_path, "r", encoding="utf8") as transcript_file:
            transcript = transcript_file.read().replace("\n", "")
        transcript = list(filter(None, [self.labels.get(x) for x in list(transcript)]))
        return transcript


class ASRDataset(LoadAudioAndTranscript):
    """
    Create ASRDataset.

    Args:
        audio_conf: Config containing the sample rate, window and the window length/stride in seconds.
        manifest_filepath (str): Manifest_file path.
        labels (list): List containing all the possible characters to map to.
        normalize: Apply standard mean and deviation Normalization to audio tensor.
        batch_size (int): Dataset batch size. Default: 32.
    """

    def __init__(
            self,
            audio_conf=None,
            manifest_filepath="",
            labels=None,
            normalize=False,
            batch_size=32,
            is_training=True,
    ):
        with open(manifest_filepath) as f_i:
            ids = f_i.readlines()

        ids = [x.strip().split(",") for x in ids]
        self.is_training = is_training
        self.ids = ids
        self.blank_id = int(labels.index("_"))
        self.bins = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]
        if len(self.ids) % batch_size != 0:
            self.bins = self.bins[:-1]
            self.bins.append(ids[-batch_size:])
        self.size = len(self.bins)
        self.batch_size = batch_size
        self.labels_map = {labels[i]: i for i in range(len(labels))}
        super(ASRDataset, self).__init__(audio_conf, normalize, self.labels_map)

    def __getitem__(self, index):
        batch_idx = self.bins[index]
        batch_size = len(batch_idx)
        batch_spect, batch_script, target_indices = [], [], []

        for data in batch_idx:
            audio_path, transcript_path = data[0], data[1]
            audio = self.load_audio(audio_path)
            transcript = self.parse_transcript(transcript_path)
            batch_spect.append(audio)
            batch_script.append(transcript)

        targets = []
        for k, scripts_ in zip(range(batch_size), batch_script):
            targets.extend(scripts_)
            for m in range(len(scripts_)):
                target_indices.append([k, m])

        return (
            batch_spect,
            np.array(target_indices, dtype=np.int64),
            np.array(targets, dtype=np.int32),
        )

    def __len__(self):
        return self.size


class DistributedSampler:
    """
    Function to distribute and shuffle sample.
    """

    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_len = len(self.dataset)
        self.num_samplers = int(math.ceil(self.dataset_len * 1.0 / self.group_size))
        self.total_size = self.num_samplers * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xFFFFFFFF
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_len).tolist()
        else:
            indices = list(range(self.dataset_len))

        indices += indices[: (self.total_size - len(indices))]
        indices = indices[self.rank :: self.group_size]
        return iter(indices)

    def __len__(self):
        return self.num_samplers


def create_dataset(
        audio_conf,
        manifest_filepath,
        labels,
        normalize,
        batch_size,
        train_mode=True,
        rank=None,
        group_size=None,
):
    """
    Create train dataset.

    Args:
        audio_conf: Config containing the sample rate, window and the window length/stride in seconds.
        manifest_filepath (str): Manifest_file path.
        labels (list): List containing all the possible characters to map to.
        normalize: Apply standard mean and deviation Normalization to audio tensor.
        train_mode (bool): Whether dataset is use for train or eval. Default: True.
        batch_size (int): Dataset batch size
        rank (int): The shard ID within num_shards. Default: None.
        group_size (int): Number of shards that the dataset should be divided into. Default: None.

    Returns:
        Dataset.
    """

    dataset = ASRDataset(
        audio_conf=audio_conf,
        manifest_filepath=manifest_filepath,
        labels=labels,
        normalize=normalize,
        batch_size=batch_size,
        is_training=train_mode,
    )

    sampler = DistributedSampler(dataset, rank, group_size, shuffle=False)

    dataset = de.GeneratorDataset(
        dataset, ["inputs", "target_indices", "label_values"], sampler=sampler
    )
    dataset = dataset.repeat(1)
    return dataset
