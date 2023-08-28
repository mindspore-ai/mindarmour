"""
The short-time Fourier transform (STFT).
"""

import numpy as np
import mindspore as ms
from mindspore import nn, ops
import librosa


class DFTBase(nn.Cell):
    """
    Base class for DFT and IDFT matrix.
    """

    def dft_matrix(self, num):
        """
        The discrete Fourier transform (DFT) matrix.
        """
        (x, y) = np.meshgrid(np.arange(num), np.arange(num))
        omega = np.exp(-2 * np.pi * 1j / num)
        w = np.power(omega, x * y)
        return w

    def idft_matrix(self, num):
        """
        The inverse discrete Fourier transform (IDFT) matrix.
        """
        (x, y) = np.meshgrid(np.arange(num), np.arange(num))
        omega = np.exp(2 * np.pi * 1j / num)
        w = np.power(omega, x * y)
        return w


class STFT(DFTBase):
    """
    The short-time Fourier transform (STFT).
    """

    def __init__(
            self,
            n_fft=2048,
            hop_length=None,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="reflect",
    ):
        """
        Implementation of STFT with Conv1d. The function has the same output
        of librosa.core.stft.
        """
        super(STFT, self).__init__()

        assert pad_mode in ["constant", "reflect"]

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if self.win_length is None:
            self.win_length = n_fft

        # Set the default hop, if it's not already specified
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)

        # Pad the window out to n_fft size
        fft_window = librosa.util.pad_center(fft_window, n_fft)

        # DFT & IDFT matrix
        self.w = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=n_fft,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            group=1,
            has_bias=False,
        )

        self.conv_imag = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=n_fft,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            group=1,
            has_bias=False,
        )

        conv_real_weight = ms.Parameter(
            np.float32(np.real(self.w[:, 0:out_channels] * fft_window[:, None])).T,
            requires_grad=False,
        )[:, None, None, :]
        self.conv_real.weight = conv_real_weight

        self.conv_imag.weight = ms.Parameter(
            np.float32(np.imag(self.w[:, 0:out_channels] * fft_window[:, None])).T,
            requires_grad=False,
        )[:, None, None, :]

    def construct(self, input_x):
        """
        input: (batch_size, data_length).
        Returns:
          real: (batch_size, n_fft // 2 + 1, time_steps).
          imag: (batch_size, n_fft // 2 + 1, time_steps).
        """

        out = input_x[:, None, :]

        if self.center:
            out = ops.pad(
                out, padding=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode
            )

        real = self.conv_real(out)
        imag = self.conv_imag(out)

        real = real[:, None, :, :].transpose((0, 1, 3, 2))
        imag = imag[:, None, :, :].transpose((0, 1, 3, 2))

        return real, imag


def magphase(real, imag):
    """
    Separate a complex-valued spectrogram into its magnitude and phase components.
    """
    mag = (real**2 + imag**2) ** 0.5
    cos = real / ops.clip_by_value(mag, 1e-10, np.inf)
    sin = imag / ops.clip_by_value(mag, 1e-10, np.inf)
    return mag, cos, sin


def ms_spectrogram(sound, ms_stft):
    """
    Convert sound to spectrogram.
    """
    real, imag = ms_stft(sound)
    mag, _, _ = magphase(real, imag)
    mag = ops.log1p(mag)

    mean = ops.mean(mag, axis=[1, 2, 3], keep_dims=True)
    std = ops.sqrt(ops.mean(ops.abs(mag - mean) ** 2))

    mag = mag - mean
    mag = mag / std

    mag = ops.permute(mag, (0, 1, 3, 2))
    return mag
