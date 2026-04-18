"""
Ensemble dataset classes for SHRED training.

Provides:
  - EnsembleSeq2SeqDataset: variable-length sequences with initial padding
  - EnsembleFrameDataset: time-delay embedded frame samples
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Optional
from sklearn.preprocessing import StandardScaler


class EnsembleSeq2SeqDataset:
    """
    Wraps multiple spatiotemporal grids for Seq2Seq SHRED training.

    Prepends L copies of the first frame to each sequence ("initial padding"),
    teaching the encoder that a constant initial signal indicates the system's
    starting state.  Handles variable-length sequences via padding to max length.
    """

    def __init__(self, sequences_in: List[np.ndarray], sensor_locs: np.ndarray,
                 initial_pad: int = 0, fit: bool = True,
                 sensor_extract_fn=None):
        """
        Args:
            sequences_in: list of (T_i, H, W) or (T_i, C, H, W) arrays
            sensor_locs: (n_sensors, 2) array of (row, col) positions
            initial_pad: number of copies of frame-0 to prepend
            fit: whether to fit new scalers
            sensor_extract_fn: optional callable (grid, sensor_locs) -> (T, n_sensors).
                Defaults to direct grid[:, r, c] extraction for scalar fields.
        """
        self.sequences = sequences_in
        self.sensor_locs = sensor_locs
        self.n_sensors = len(sensor_locs)

        s0 = sequences_in[0]
        self.spatial_shape = s0.shape[1:]  # (H, W) or (C, H, W)
        self.state_dim = int(np.prod(self.spatial_shape))

        self.T_originals = [s.shape[0] for s in sequences_in]

        if sensor_extract_fn is None:
            sensor_extract_fn = _default_sensor_extract
        self._extract = sensor_extract_fn

        # Prepend initial padding
        self.initial_pad = 0
        self.sequences_padded = list(sequences_in)
        self.T_padded = [s.shape[0] for s in self.sequences_padded]

        # Extract sensor series and flatten states
        self.sensor_series, self.states = [], []
        for seq in self.sequences_padded:
            self.sensor_series.append(self._extract(seq, sensor_locs))
            self.states.append(seq.reshape(seq.shape[0], -1))

        # Scalers
        self.scaler_sensors = StandardScaler()
        self.scaler_states = StandardScaler()
        if fit:
            self.scaler_sensors.fit(np.concatenate(self.sensor_series, axis=0))
            self.scaler_states.fit(np.concatenate(self.states, axis=0))

        self.sensor_series_scaled = [self.scaler_sensors.transform(s) for s in self.sensor_series]
        self.states_scaled = [self.scaler_states.transform(s) for s in self.states]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (jnp.array(self.sensor_series_scaled[idx], dtype=jnp.float32),
                jnp.array(self.states_scaled[idx], dtype=jnp.float32))

    def get_batches(self, batch_size, shuffle=True, rng=None):
        n = len(self)
        indices = np.arange(n)
        if shuffle and rng is not None:
            indices = np.array(jax.random.permutation(rng, indices))

        batches = []
        for i in range(0, n, batch_size):
            bidx = indices[i:i + batch_size]
            if len(bidx) == 0:
                continue
            max_T = max(self.sensor_series_scaled[j].shape[0] for j in bidx)
            xb, yb = [], []
            for j in bidx:
                s, st = self.sensor_series_scaled[j], self.states_scaled[j]
                T = s.shape[0]
                if T < max_T:
                    ps = np.zeros((max_T, self.n_sensors)); ps[:T] = s; ps[T:] = s[-1]
                    pst = np.zeros((max_T, self.state_dim)); pst[:T] = st; pst[T:] = st[-1]
                else:
                    ps, pst = s, st
                xb.append(ps); yb.append(pst)
            batches.append((jnp.array(np.stack(xb), dtype=jnp.float32),
                            jnp.array(np.stack(yb), dtype=jnp.float32)))
        return batches


class EnsembleFrameDataset:
    """
    Wraps multiple spatiotemporal grids for frame-by-frame SHRED training.

    Uses time-delay embedding: input at time t is sensor_readings[t-lags+1:t+1],
    target is the full spatial state at time t.
    """

    def __init__(self, sequences_in: List[np.ndarray], sensor_locs: np.ndarray,
                 lags: int = 5, fit: bool = True,
                 sensor_extract_fn=None):
        self.sequences = sequences_in
        self.sensor_locs = sensor_locs
        self.n_sensors = len(sensor_locs)
        self.lags = lags

        s0 = sequences_in[0]
        self.spatial_shape = s0.shape[1:]
        self.state_dim = int(np.prod(self.spatial_shape))

        self.T_originals = [s.shape[0] for s in sequences_in]

        if sensor_extract_fn is None:
            sensor_extract_fn = _default_sensor_extract
        self._extract = sensor_extract_fn

        all_X, all_Y = [], []
        for seq in sequences_in:
            T = seq.shape[0]
            sens = self._extract(seq, sensor_locs)
            N = T - lags
            if N <= 0:
                continue
            for i in range(N):
                all_X.append(sens[i:i + lags])
                all_Y.append(seq[i + lags - 1].ravel())

        self.X = np.array(all_X, dtype=np.float32)
        self.Y = np.array(all_Y, dtype=np.float32)

        self.scaler_sensors = StandardScaler()
        self.scaler_states = StandardScaler()
        if fit:
            self.scaler_sensors.fit(self.X.reshape(-1, self.n_sensors))
            self.scaler_states.fit(self.Y)

        X_flat = self.X.reshape(-1, self.n_sensors)
        self.X_scaled = self.scaler_sensors.transform(X_flat).reshape(self.X.shape)
        self.Y_scaled = self.scaler_states.transform(self.Y)

        self.initial_pad = 0  # compatibility attribute

    def __len__(self):
        return len(self.X)

    def get_batches(self, batch_size, shuffle=True, rng=None):
        n = len(self)
        indices = np.arange(n)
        if shuffle and rng is not None:
            indices = np.array(jax.random.permutation(rng, indices))

        batches = []
        for i in range(0, n, batch_size):
            bidx = indices[i:i + batch_size]
            if len(bidx) == 0:
                continue
            batches.append((jnp.array(self.X_scaled[bidx], dtype=jnp.float32),
                            jnp.array(self.Y_scaled[bidx], dtype=jnp.float32)))
        return batches


def _default_sensor_extract(grid, sensor_locs):
    """Default sensor extraction: grid[:, r, c] for scalar fields (T, H, W)."""
    return np.stack([grid[:, r, c] for r, c in sensor_locs], axis=1)
