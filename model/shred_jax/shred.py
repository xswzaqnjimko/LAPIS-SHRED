"""
SHRED (SHallow REcurrent Decoder) — JAX/Flax implementation.

Core models for reconstructing full spatiotemporal fields from sparse sensor
measurements.  Based on Williams, Kutz & Brunton (2023).

Provides:
  - BidirectionalLSTM: multi-layer bidirectional encoder
  - MLPDecoder: LayerNorm + GELU feedforward decoder
  - FrameSHRED: frame-by-frame reconstruction via time-delay embedding
  - Seq2SeqSHRED: sequence-to-sequence full-trajectory reconstruction
  - Loss functions: weighted_mse_loss, create_active_weights
  - Metrics: compute_metrics (RMSE, MAE, SSIM, IoU, correlation)
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from typing import Tuple, List, Optional, Dict
from functools import partial
from scipy.ndimage import uniform_filter


# Models

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM encoder with stacked layers and inter-layer dropout."""
    hidden_size: int
    num_layers: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            outputs: (batch, seq_len, hidden_size * 2)
            h_final: (batch, hidden_size * 2)
        """
        for layer_idx in range(self.num_layers):
            lstm_fwd = nn.RNN(
                nn.LSTMCell(features=self.hidden_size),
                return_carry=True, name=f'lstm_fwd_{layer_idx}')
            carry_fwd, outputs_fwd = lstm_fwd(x)
            h_fwd = carry_fwd[0]

            x_rev = jnp.flip(x, axis=1)
            lstm_bwd = nn.RNN(
                nn.LSTMCell(features=self.hidden_size),
                return_carry=True, name=f'lstm_bwd_{layer_idx}')
            carry_bwd, outputs_bwd_rev = lstm_bwd(x_rev)
            outputs_bwd = jnp.flip(outputs_bwd_rev, axis=1)
            h_bwd = carry_bwd[0]

            x = jnp.concatenate([outputs_fwd, outputs_bwd], axis=-1)
            if layer_idx < self.num_layers - 1 and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        h_final = jnp.concatenate([h_fwd, h_bwd], axis=-1)
        return x, h_final


class MLPDecoder(nn.Module):
    """MLP decoder with LayerNorm and GELU activation."""
    layer_sizes: Tuple[int, ...]
    output_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        for i, size in enumerate(self.layer_sizes):
            x = nn.Dense(size, name=f'dense_{i}')(x)
            x = nn.LayerNorm(name=f'ln_{i}')(x)
            x = nn.gelu(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        return nn.Dense(self.output_dim, name='output')(x)


class FrameSHRED(nn.Module):
    """Frame-by-frame SHRED: BiLSTM encoder + MLP decoder per time step."""
    n_sensors: int
    hidden_size: int = 128
    num_layers: int = 2
    decoder_layers: Tuple[int, ...] = (256, 256, 128)
    state_dim: int = 9760
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        encoder = BidirectionalLSTM(
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout_rate=self.dropout_rate if self.num_layers > 1 else 0.0)
        _, h_final = encoder(x, train=train)
        h_norm = nn.LayerNorm(name='h_norm')(h_final)
        decoder = MLPDecoder(
            layer_sizes=self.decoder_layers, output_dim=self.state_dim,
            dropout_rate=self.dropout_rate)
        return decoder(h_norm, train=train)


class Seq2SeqSHRED(nn.Module):
    """Sequence-to-sequence SHRED: full trajectory reconstruction."""
    n_sensors: int
    hidden_size: int = 256
    num_layers: int = 2
    state_dim: int = 9760
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        batch_size, seq_len, _ = x.shape
        encoder = BidirectionalLSTM(
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout_rate=self.dropout_rate if self.num_layers > 1 else 0.0)
        enc_outputs, _ = encoder(x, train=train)

        lstm_dec = nn.RNN(nn.LSTMCell(features=self.hidden_size), name='lstm_decoder')
        dec_outputs = lstm_dec(enc_outputs)

        dec_flat = dec_outputs.reshape(batch_size * seq_len, self.hidden_size)
        h = nn.Dense(self.hidden_size * 2, name='mlp1')(dec_flat)
        h = nn.LayerNorm(name='ln1')(h); h = nn.gelu(h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)
        h = nn.Dense(self.hidden_size * 2, name='mlp2')(h)
        h = nn.LayerNorm(name='ln2')(h); h = nn.gelu(h)
        output = nn.Dense(self.state_dim, name='output')(h)
        return output.reshape(batch_size, seq_len, self.state_dim)


# Loss functions

def weighted_mse_loss(pred, target, weights):
    sq_error = (pred - target) ** 2
    if pred.ndim == 2:
        weighted_error = sq_error * weights[None, :]
    else:
        weighted_error = sq_error * weights[None, None, :]
    return jnp.mean(weighted_error)


def create_active_weights(active_mask, active_weight=20.0):
    weights = np.ones(active_mask.size, dtype=np.float32)
    weights[active_mask.ravel()] = active_weight
    weights = weights / weights.mean()
    return jnp.array(weights)


# Metrics

def compute_metrics(pred, target, active_mask, pred_scaled=None, target_scaled=None):
    """Compute RMSE, MAE, SSIM, correlation, IoU for reconstruction quality."""
    if pred.ndim == 3:
        T = pred.shape[0]
        if pred_scaled is not None and target_scaled is not None:
            metrics_list = [
                compute_metrics(pred[t], target[t], active_mask, pred_scaled[t], target_scaled[t])
                for t in range(T)]
        else:
            metrics_list = [compute_metrics(pred[t], target[t], active_mask) for t in range(T)]
        return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

    pred_flat = pred.ravel()
    target_flat = target.ravel()
    active_flat = active_mask.ravel()

    rmse_full = np.sqrt(np.mean((pred_flat - target_flat) ** 2))

    if active_flat.sum() > 0:
        pred_active = pred_flat[active_flat]
        target_active = target_flat[active_flat]
        rmse_active = np.sqrt(np.mean((pred_active - target_active) ** 2))
        mae_active = np.mean(np.abs(pred_active - target_active))
        corr_active = (np.corrcoef(pred_active, target_active)[0, 1]
                       if np.std(target_active) > 1e-6 else 0.0)
    else:
        rmse_active = mae_active = corr_active = 0.0

    if pred_scaled is not None and target_scaled is not None:
        ps, ts = pred_scaled.ravel(), target_scaled.ravel()
        rmse_full_scaled = np.sqrt(np.mean((ps - ts) ** 2))
        rmse_active_scaled = (np.sqrt(np.mean((ps[active_flat] - ts[active_flat]) ** 2))
                              if active_flat.sum() > 0 else 0.0)
    else:
        rmse_full_scaled = rmse_active_scaled = None

    def _ssim(x, y):
        L = max(x.max() - x.min(), y.max() - y.min(), 1e-6)
        c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
        mx, my = uniform_filter(x, 5), uniform_filter(y, 5)
        sx2 = uniform_filter(x ** 2, 5) - mx ** 2
        sy2 = uniform_filter(y ** 2, 5) - my ** 2
        sxy = uniform_filter(x * y, 5) - mx * my
        return np.mean((2*mx*my + c1) * (2*sxy + c2) / ((mx**2 + my**2 + c1) * (sx2 + sy2 + c2)))

    ssim = _ssim(pred.reshape(active_mask.shape), target.reshape(active_mask.shape))

    pred_extent = np.abs(pred_flat) > 0.1
    target_extent = np.abs(target_flat) > 0.1
    intersection = np.sum(pred_extent & target_extent)
    union = np.sum(pred_extent | target_extent)
    iou = intersection / max(union, 1)

    result = {
        'rmse_active': rmse_active, 'mae_active': mae_active,
        'corr_active': corr_active, 'rmse_full': rmse_full,
        'ssim': ssim, 'iou': iou,
    }
    if rmse_full_scaled is not None:
        result['rmse_full_scaled'] = rmse_full_scaled
        result['rmse_active_scaled'] = rmse_active_scaled
    return result
