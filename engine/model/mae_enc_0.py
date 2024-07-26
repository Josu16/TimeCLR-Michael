import copy
import numpy as np
import torch
import torch.nn as nn


class MAEEncoder(nn.Module):
    def __init__(self, encoder, mask_ratio=0.15):
        """
        Mask AutoEncoder (MAE) model for self-supervised learning.
        
        Args:
            encoder (Module): The base encoder
            mask_ratio (float, optional): The ratio of time steps to mask.
                Default: 0.15.

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`.
            - Output: :math:`(N, C_{out})`.
        """
        super(MAEEncoder, self).__init__()

        self.pretrain_name = 'mae'
        self.encoder = copy.deepcopy(encoder)
        self.mask_ratio = mask_ratio

        self.out_dim = self.encoder.out_dim
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, ts, normalize=True, to_numpy=False, is_augment=False):
        if not is_augment:
            ts_emb = self.encoder.encode(
                ts, normalize=normalize, to_numpy=to_numpy)
            return ts_emb

        ts_masked, mask = self._mask_time_steps(ts)
        ts_emb = self.encoder.encode(
            ts_masked, normalize=normalize, to_numpy=to_numpy)
        # ts_reconstructed = self.decode(ts_emb)
        # print("mask shape otra vez---", mask.shape)
        return ts_masked, mask

    # def _mask_time_steps(self, ts):
    #     ts_masked = copy.deepcopy(ts)
    #     mask = torch.rand(ts_masked.shape) < self.mask_ratio
    #     ts_masked[mask] = 0  # Masked time steps set to zero
    #     return ts_masked, mask

    def _mask_time_steps(self, ts):
        ts_masked = ts.copy()
        samples, channels, steps = ts.shape
        result = np.empty((0, channels, steps))
        # Enmascarado por fila
        for i in range(samples):
            random_values = np.random.rand(channels, steps)
            mask = random_values < self.mask_ratio
            ts_masked[i][mask] = 0  # Asumiendo que 0 es el valor de enmascarado

            arr = mask.reshape(1, channels, steps)
            result = np.concatenate((result, arr), axis=0)

        # print("máscara shape--------", result.shape)
        # print("ts_masked shape--------", ts_masked.shape)
        return ts_masked, result

    # def _mask_time_steps(self, ts):
    #     ts_masked = copy.deepcopy(ts)
    #     mask = np.random.rand(*ts_masked.shape) < self.mask_ratio
    #     ts_masked[mask] = 0  # Masked time steps set to zero
    #     return ts_masked, mask

    def encode(self, ts, normalize=True, to_numpy=False):
        ts_emb = self.encoder.encode(
            ts, normalize=normalize, to_numpy=to_numpy)
        return ts_emb
    # def decode(self, z):
    #         z = z.unsqueeze(1)  # Añadimos una dimensión para el LSTM
    #         output, _ = self.decoder_lstm(z)
    #         output = self.output_layer(output.squeeze(1))
    #         return output.view(output.size(0), 1, -1)  # Ajustamos la salida a la forma (N, C, L)
