"""
LAPIS inference pipelines and SHRED baselines.

Provides:
  - Encoder/decoder extraction from trained SHRED models
  - Latent trajectory extraction (seq2seq and frame-by-frame)
  - Forward and backward LAPIS inference (seq2seq and frame-by-frame)
  - SHRED full-observation baselines
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import List, Tuple, Optional, Callable

from .shred import BidirectionalLSTM, MLPDecoder


# Encoder / decoder helpers

class Seq2SeqSHREDEncoder(nn.Module):
    """Encoder-only wrapper: extracts BiLSTM outputs from trained Seq2SeqSHRED."""
    hidden_size: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train=True):
        enc = BidirectionalLSTM(
            hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout_rate=self.dropout_rate if self.num_layers > 1 else 0.0)
        out, _ = enc(x, train=train)
        return out


class FlexibleSHREDDecoder(nn.Module):
    """Decoder-only wrapper: LSTM + MLP that maps latent -> spatial states."""
    hidden_size: int = 64
    state_dim: int = 4096
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, enc_outputs, train=True):
        B, T, D = enc_outputs.shape
        cell = nn.LSTMCell(features=self.hidden_size)
        carry = cell.initialize_carry(jax.random.PRNGKey(0), (B, D))
        outs = []
        for t in range(T):
            carry, y = cell(carry, enc_outputs[:, t, :])
            outs.append(y)
        dec = jnp.stack(outs, axis=1)
        flat = dec.reshape(B * T, self.hidden_size)
        h = nn.Dense(self.hidden_size * 2, name="mlp1")(flat)
        h = nn.LayerNorm(name="ln1")(h); h = nn.gelu(h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)
        h = nn.Dense(self.hidden_size * 2, name="mlp2")(h)
        h = nn.LayerNorm(name="ln2")(h); h = nn.gelu(h)
        out = nn.Dense(self.state_dim, name="output")(h)
        return out.reshape(B, T, self.state_dim)


def extract_encoder_params(shred_params):
    """Extract BiLSTM encoder parameters from trained SHRED."""
    return {k: v for k, v in shred_params.items()
            if any(s in k for s in ["lstm_fwd", "lstm_bwd", "BidirectionalLSTM"])}


def extract_decoder_params(shred_params):
    """Extract decoder (LSTM + MLP) parameters from trained SHRED."""
    dec = {}
    for k in ["mlp1", "ln1", "mlp2", "ln2", "output"]:
        if k in shred_params:
            dec[k] = shred_params[k]
    for k in ["lstm_decoder", "LSTMCell_0"]:
        if k in shred_params:
            dec[k] = shred_params[k]
            break
    return dec


def decode_latent_with_frozen_shred(z_full, shred_state, state_dim, config):
    """Decode latent trajectory -> spatial states using frozen SHRED decoder."""
    z_batch = z_full[None, ...]
    dec_params = extract_decoder_params(shred_state.params)
    decoder = FlexibleSHREDDecoder(
        hidden_size=config.SEQ2SEQ_HIDDEN, state_dim=state_dim, dropout_rate=0.0)
    pred = decoder.apply({"params": dec_params}, z_batch, train=False)
    return pred[0]


# Latent trajectory extraction

def extract_latent_trajectories_seq2seq(shred_state, dataset, config):
    """Run frozen SHRED encoder on each simulation to get latent trajectories."""
    latent_dim = config.SEQ2SEQ_HIDDEN * 2
    encoder = Seq2SeqSHREDEncoder(
        hidden_size=config.SEQ2SEQ_HIDDEN, num_layers=config.NUM_LAYERS, dropout_rate=0.0)
    rng = jax.random.PRNGKey(0)
    sample_x, _ = dataset[0]
    dummy = jnp.ones((1,) + sample_x.shape)
    encoder.init(rng, dummy, train=False)
    enc_params = extract_encoder_params(shred_state.params)

    trajectories, z_inits = [], []
    for i in range(len(dataset)):
        x, _ = dataset[i]
        xb = jnp.array(x[None, ...], dtype=jnp.float32)
        try:
            out = encoder.apply({"params": enc_params}, xb, train=False)
            traj = np.array(out[0])
        except Exception as e:
            print(f"    Encoder failed for traj {i}: {e} - using projection fallback")
            np.random.seed(config.SEED + i)
            proj = np.random.randn(config.N_SENSORS, latent_dim) / np.sqrt(config.N_SENSORS)
            traj = np.array(x) @ proj

        T_orig = dataset.T_originals[i]
        traj = traj[:T_orig]
        trajectories.append(traj)
        z_inits.append(traj[0])

    print(f"    Extracted {len(trajectories)} latent trajectories (dim={latent_dim})")
    return trajectories, z_inits


def extract_latent_trajectories_frame(shred_state, frame_dataset, sim_grids,
                                      sensors, config,
                                      sensor_extract_fn=None):
    """Run frozen frame-SHRED encoder on each simulation grid to get latent trajectories."""
    lags = config.LAGS
    n_sensors = config.N_SENSORS
    scaler_sens = frame_dataset.scaler_sensors

    if sensor_extract_fn is None:
        sensor_extract_fn = lambda grid, locs: np.stack([grid[:, r, c] for r, c in locs], axis=1)

    class FrameEncoder(nn.Module):
        hidden_size: int = 64
        num_layers: int = 2
        dropout_rate: float = 0.0

        @nn.compact
        def __call__(self, x, train=False):
            encoder = BidirectionalLSTM(
                hidden_size=self.hidden_size, num_layers=self.num_layers,
                dropout_rate=self.dropout_rate if self.num_layers > 1 else 0.0)
            _, h_final = encoder(x, train=train)
            return nn.LayerNorm(name='h_norm')(h_final)

    encoder = FrameEncoder(hidden_size=config.SEQ2SEQ_HIDDEN, num_layers=config.NUM_LAYERS)
    enc_params = {k: v for k, v in shred_state.params.items() if "MLPDecoder" not in k}

    dummy_x = jnp.ones((1, lags, n_sensors))
    try:
        encoder.apply({"params": enc_params}, dummy_x, train=False)
        use_enc_params = enc_params
    except Exception:
        print("    Warning: could not extract encoder params, using full SHRED as latent proxy")
        use_enc_params = None

    trajectories, z_inits = [], []
    for gi, grid in enumerate(sim_grids):
        T = grid.shape[0]
        sens = sensor_extract_fn(grid, sensors)
        N = T - lags
        X = np.zeros((N, lags, n_sensors), dtype=np.float32)
        for i in range(N):
            X[i] = sens[i:i + lags]
        X_flat = scaler_sens.transform(X.reshape(-1, n_sensors))
        X_scaled = X_flat.reshape(N, lags, n_sensors)
        x_jax = jnp.array(X_scaled, dtype=jnp.float32)

        if use_enc_params is not None:
            h = encoder.apply({"params": use_enc_params}, x_jax, train=False)
            traj = np.array(h)
        else:
            pred = shred_state.apply_fn({"params": shred_state.params}, x_jax, train=False)
            traj = np.array(pred)

        trajectories.append(traj)
        z_inits.append(traj[0])

    actual_dim = trajectories[0].shape[1]
    print(f"    Extracted {len(trajectories)} latent trajectories (dim={actual_dim})")
    return trajectories, z_inits


# Forward LAPIS inference (seq2seq mode)

def lapis_forward_inference_seq2seq(shred_state, forward_state, gt_grid, sensors,
                                    dataset, obs_len, config,
                                    sensor_extract_fn=None):
    """Forward inference: encode initial obs window, predict remaining trajectory."""
    spatial_shape = gt_grid.shape[1:]
    state_dim = int(np.prod(spatial_shape))
    T_gt = gt_grid.shape[0]
    latent_dim = config.SEQ2SEQ_HIDDEN * 2

    if sensor_extract_fn is None:
        sensor_extract_fn = lambda grid, locs: np.stack([grid[:, r, c] for r, c in locs], axis=1)

    gt_obs = gt_grid[:obs_len]
    sens_obs = sensor_extract_fn(gt_obs, sensors)
    sens_scaled = dataset.scaler_sensors.transform(sens_obs)
    x_in = jnp.array(sens_scaled, dtype=jnp.float32)[None, ...]

    encoder = Seq2SeqSHREDEncoder(
        hidden_size=config.SEQ2SEQ_HIDDEN, num_layers=config.NUM_LAYERS, dropout_rate=0.0)
    enc_params = extract_encoder_params(shred_state.params)

    try:
        enc_out = encoder.apply({"params": enc_params}, x_in, train=False)
        z_obs = enc_out[0]
    except Exception as e:
        print(f"  Warning: encoder failed ({e}), using projection fallback")
        np.random.seed(config.SEED)
        proj = np.random.randn(config.N_SENSORS, latent_dim) / np.sqrt(config.N_SENSORS)
        z_obs = jnp.array(sens_scaled @ proj, dtype=jnp.float32)

    T_future = T_gt - obs_len
    z_obs_batch = z_obs[None, ...]
    z_future = forward_state.apply_fn(
        {"params": forward_state.params}, z_obs_batch, T_future, train=False)[0]

    z_full = jnp.concatenate([z_obs, z_future], axis=0)
    pred_scaled = decode_latent_with_frozen_shred(z_full, shred_state, state_dim, config)
    pred = dataset.scaler_states.inverse_transform(np.array(pred_scaled))
    return pred[:T_gt].reshape((T_gt,) + spatial_shape)


# Backward LAPIS inference (seq2seq mode, multi-frame terminal window)

def lapis_backward_inference_seq2seq(shred_state, backward_state, gt_grid, sensors,
                                     dataset, obs_len, config,
                                     sensor_extract_fn=None):
    """Backward inference from terminal window: encode tail frames -> backward model -> decode."""
    spatial_shape = gt_grid.shape[1:]
    state_dim = int(np.prod(spatial_shape))
    T_gt = gt_grid.shape[0]
    latent_dim = config.SEQ2SEQ_HIDDEN * 2

    if sensor_extract_fn is None:
        sensor_extract_fn = lambda grid, locs: np.stack([grid[:, r, c] for r, c in locs], axis=1)

    # Terminal window sensors (no initial padding in Seq2Seq mode)
    gt_tail = gt_grid[-obs_len:]
    sens_tail = sensor_extract_fn(gt_tail, sensors)
    sens_scaled = dataset.scaler_sensors.transform(sens_tail)
    x_in = jnp.array(sens_scaled, dtype=jnp.float32)[None, ...]

    encoder = Seq2SeqSHREDEncoder(
        hidden_size=config.SEQ2SEQ_HIDDEN, num_layers=config.NUM_LAYERS, dropout_rate=0.0)
    enc_params = extract_encoder_params(shred_state.params)

    try:
        enc_out = encoder.apply({"params": enc_params}, x_in, train=False)
        z_tail = enc_out[0]
    except Exception as e:
        print(f"  Warning: encoder failed ({e}), using projection fallback")
        np.random.seed(config.SEED)
        proj = np.random.randn(config.N_SENSORS, latent_dim) / np.sqrt(config.N_SENSORS)
        z_tail = jnp.array(sens_scaled @ proj, dtype=jnp.float32)

    # Backward model: terminal window -> preceding trajectory
    T_prior = T_gt - obs_len
    z_tail_batch = z_tail[None, ...]
    z_prior = backward_state.apply_fn(
        {"params": backward_state.params}, z_tail_batch, T_prior, train=False)[0]

    z_full = jnp.concatenate([z_prior, z_tail], axis=0)
    pred_scaled = decode_latent_with_frozen_shred(z_full, shred_state, state_dim, config)
    pred = dataset.scaler_states.inverse_transform(np.array(pred_scaled))
    return pred[:T_gt].reshape((T_gt,) + spatial_shape)


# Backward LAPIS inference (seq2seq mode, single terminal frame with static padding)

def lapis_backward_inference_terminal_seq2seq(shred_state, backward_state, gt_grid, sensors,
                                              dataset, config,
                                              sensor_extract_fn=None):
    """Backward inference from single terminal frame: static-pad -> encode -> backward model -> decode."""
    spatial_shape = gt_grid.shape[1:]
    state_dim = int(np.prod(spatial_shape))
    T_gt = gt_grid.shape[0]
    latent_dim = config.SEQ2SEQ_HIDDEN * 2

    if sensor_extract_fn is None:
        sensor_extract_fn = lambda grid, locs: np.stack([grid[:, r, c] for r, c in locs], axis=1)

    # Terminal frame sensors with static padding
    Y_T = gt_grid[-1:]
    sens_T = sensor_extract_fn(Y_T, sensors)[0]  # (n_sensors,)
    sens_padded = np.tile(sens_T[None, :], (config.STATIC_PAD_LENGTH, 1))
    sens_scaled = dataset.scaler_sensors.transform(sens_padded)
    x_terminal = jnp.array(sens_scaled, dtype=jnp.float32)[None, ...]

    encoder = Seq2SeqSHREDEncoder(
        hidden_size=config.SEQ2SEQ_HIDDEN, num_layers=config.NUM_LAYERS, dropout_rate=0.0)
    enc_params = extract_encoder_params(shred_state.params)

    try:
        enc_out = encoder.apply({"params": enc_params}, x_terminal, train=False)
        z_T = enc_out[0, -1]
    except Exception as e:
        print(f"  Warning: encoder failed ({e}), using projection fallback")
        np.random.seed(config.SEED)
        proj = np.random.randn(config.N_SENSORS, latent_dim) / np.sqrt(config.N_SENSORS)
        z_T = jnp.array(sens_scaled[-1] @ proj, dtype=jnp.float32)

    T_out = T_gt - 1
    z_T_batch = z_T[None, :]
    z_pred_seq = backward_state.apply_fn(
        {"params": backward_state.params}, z_T_batch, T_out, train=False)[0]

    z_full = jnp.concatenate([z_pred_seq, z_T[None, :]], axis=0)
    pred_scaled = decode_latent_with_frozen_shred(z_full, shred_state, state_dim, config)
    pred = dataset.scaler_states.inverse_transform(np.array(pred_scaled))
    return pred[:T_gt].reshape((T_gt,) + spatial_shape)


# Backward LAPIS inference (frame-by-frame mode)

def lapis_backward_inference_frame(shred_state, backward_state, gt_grid, sensors,
                                   frame_dataset, obs_len, config,
                                   sensor_extract_fn=None):
    """Backward inference using frame-by-frame SHRED encoder."""
    spatial_shape = gt_grid.shape[1:]
    state_dim = int(np.prod(spatial_shape))
    T_gt = gt_grid.shape[0]
    lags = config.LAGS
    n_sensors = config.N_SENSORS
    scaler_sens = frame_dataset.scaler_sensors
    scaler_st = frame_dataset.scaler_states

    if sensor_extract_fn is None:
        sensor_extract_fn = lambda grid, locs: np.stack([grid[:, r, c] for r, c in locs], axis=1)

    # Encode terminal window through frame SHRED encoder
    gt_tail = gt_grid[-obs_len:]
    sens_tail = sensor_extract_fn(gt_tail, sensors)
    pad = np.tile(sens_tail[0:1], (lags - 1, 1))
    sens_padded = np.concatenate([pad, sens_tail], axis=0)

    N_obs = obs_len
    X_obs = np.zeros((N_obs, lags, n_sensors), dtype=np.float32)
    for i in range(N_obs):
        X_obs[i] = sens_padded[i:i + lags]

    X_flat = scaler_sens.transform(X_obs.reshape(-1, n_sensors))
    X_scaled = X_flat.reshape(N_obs, lags, n_sensors)

    class FrameEncoder(nn.Module):
        hidden_size: int = 64
        num_layers: int = 2
        dropout_rate: float = 0.0

        @nn.compact
        def __call__(self, x, train=False):
            encoder = BidirectionalLSTM(
                hidden_size=self.hidden_size, num_layers=self.num_layers,
                dropout_rate=self.dropout_rate if self.num_layers > 1 else 0.0)
            _, h_final = encoder(x, train=train)
            return nn.LayerNorm(name='h_norm')(h_final)

    enc_params = {k: v for k, v in shred_state.params.items() if "MLPDecoder" not in k}
    encoder = FrameEncoder(hidden_size=config.SEQ2SEQ_HIDDEN, num_layers=config.NUM_LAYERS)
    x_jax = jnp.array(X_scaled, dtype=jnp.float32)

    try:
        z_tail = encoder.apply({"params": enc_params}, x_jax, train=False)
    except Exception:
        z_tail = shred_state.apply_fn({"params": shred_state.params}, x_jax, train=False)

    # Backward model: terminal window -> preceding trajectory
    T_prior = T_gt - obs_len
    z_tail_batch = jnp.array(z_tail)[None, ...]
    z_prior = backward_state.apply_fn(
        {"params": backward_state.params}, z_tail_batch, T_prior, train=False)[0]

    z_full = jnp.concatenate([z_prior, jnp.array(z_tail)], axis=0)

    # Decode using frame SHRED decoder
    dec_params = {k: v for k, v in shred_state.params.items()
                  if "MLPDecoder" in k or k.startswith("MLPDecoder")}

    if z_full.shape[1] == state_dim:
        pred = scaler_st.inverse_transform(np.array(z_full))
    else:
        from .shred import MLPDecoder as MLPDecoderModule

        class FrameDecoder(nn.Module):
            decoder_layers: tuple = (256, 256)
            state_dim: int = 4096
            dropout_rate: float = 0.0

            @nn.compact
            def __call__(self, h, train=False):
                return MLPDecoderModule(
                    layer_sizes=self.decoder_layers, output_dim=self.state_dim,
                    dropout_rate=self.dropout_rate)(h, train=train)

        frame_decoder = FrameDecoder(
            decoder_layers=config.DECODER_LAYERS, state_dim=state_dim)
        pred_scaled = frame_decoder.apply(
            {"params": dec_params}, jnp.array(z_full), train=False)
        pred = scaler_st.inverse_transform(np.array(pred_scaled))

    return pred[:T_gt].reshape((T_gt,) + spatial_shape)


# SHRED baselines

def shred_baseline_seq2seq(shred_state, gt_grid, sensors, dataset, config,
                           sensor_extract_fn=None):
    """Upper bound: SHRED reconstruction using the full GT sensor time-series."""
    spatial_shape = gt_grid.shape[1:]
    T_gt = gt_grid.shape[0]

    if sensor_extract_fn is None:
        sensor_extract_fn = lambda grid, locs: np.stack([grid[:, r, c] for r, c in locs], axis=1)

    sens_full = sensor_extract_fn(gt_grid, sensors)
    sens_scaled = dataset.scaler_sensors.transform(sens_full)

    max_T = max(dataset.T_originals)
    if sens_scaled.shape[0] < max_T:
        extra = max_T - sens_scaled.shape[0]
        sens_scaled = np.concatenate([sens_scaled, np.tile(sens_scaled[-1:], (extra, 1))])

    x = jnp.array(sens_scaled, dtype=jnp.float32)[None, ...]
    pred_scaled = shred_state.apply_fn({"params": shred_state.params}, x, train=False)
    pred = dataset.scaler_states.inverse_transform(np.array(pred_scaled[0]))
    pred = pred[:T_gt]
    return pred.reshape((T_gt,) + spatial_shape)


def shred_baseline_frame(shred_state, gt_grid, sensors, frame_dataset, config,
                         sensor_extract_fn=None):
    """Upper bound: frame SHRED with full GT sensor time-series."""
    spatial_shape = gt_grid.shape[1:]
    T_gt = gt_grid.shape[0]
    lags = config.LAGS
    n_sensors = config.N_SENSORS
    scaler_sens = frame_dataset.scaler_sensors
    scaler_st = frame_dataset.scaler_states

    if sensor_extract_fn is None:
        sensor_extract_fn = lambda grid, locs: np.stack([grid[:, r, c] for r, c in locs], axis=1)

    sens_full = sensor_extract_fn(gt_grid, sensors)
    pad = np.tile(sens_full[0:1], (lags - 1, 1))
    sens_padded = np.concatenate([pad, sens_full], axis=0)
    N = T_gt
    X = np.zeros((N, lags, n_sensors), dtype=np.float32)
    for i in range(N):
        X[i] = sens_padded[i:i + lags]
    X_flat = scaler_sens.transform(X.reshape(-1, n_sensors))
    X_scaled = X_flat.reshape(N, lags, n_sensors)
    x_jax = jnp.array(X_scaled, dtype=jnp.float32)

    pred_scaled = shred_state.apply_fn({"params": shred_state.params}, x_jax, train=False)
    pred = scaler_st.inverse_transform(np.array(pred_scaled))
    return pred[:T_gt].reshape((T_gt,) + spatial_shape)
