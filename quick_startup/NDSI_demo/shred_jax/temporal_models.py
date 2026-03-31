"""
Temporal dynamics models for LAPIS inference.

Provides:
  - ForwardFromWindow: initial latent window -> future latent trajectory
  - BackwardFromWindow: terminal latent window -> preceding latent trajectory
  - BackwardFromTerminal: single terminal latent -> full preceding trajectory
"""

import jax.numpy as jnp
import flax.linen as nn
from .shred import BidirectionalLSTM


class ForwardFromWindow(nn.Module):
    """Predict future latent trajectory from an initial observed window."""
    latent_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, z_init_seq: jnp.ndarray, T_out: int, train: bool = True):
        """
        Args:
            z_init_seq: (B, W, latent_dim) — observed initial latent window
            T_out: number of future frames to predict
        Returns:
            (B, T_out, latent_dim)
        """
        B, W, D = z_init_seq.shape

        enc = BidirectionalLSTM(
            hidden_size=self.hidden_dim, num_layers=self.num_layers,
            dropout_rate=self.dropout_rate if self.num_layers > 1 else 0.0)
        _, h_init = enc(z_init_seq, train=train)

        h_rep = jnp.tile(h_init[:, None, :], (1, T_out, 1))
        pos = jnp.tile(jnp.linspace(0, 1, T_out)[None, :, None], (B, 1, 1))
        x = jnp.concatenate([h_rep, pos], axis=-1)
        x = nn.Dense(self.latent_dim, name="input_proj")(x)

        enc2 = BidirectionalLSTM(hidden_size=self.hidden_dim, num_layers=1, dropout_rate=0.0)
        enc2_out, _ = enc2(x, train=train)

        lstm_dec = nn.RNN(nn.LSTMCell(features=self.hidden_dim), name="lstm_decoder")
        dec_out = lstm_dec(enc2_out)

        flat = dec_out.reshape(B * T_out, self.hidden_dim)
        h = nn.Dense(self.hidden_dim * 2, name="mlp1")(flat)
        h = nn.LayerNorm(name="ln1")(h); h = nn.gelu(h)
        out = nn.Dense(self.latent_dim, name="output")(h)
        return out.reshape(B, T_out, self.latent_dim)


class BackwardFromWindow(nn.Module):
    """Predict preceding latent trajectory from a terminal observed window."""
    latent_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, z_window: jnp.ndarray, T_out: int, train: bool = True):
        """
        Args:
            z_window: (B, W, latent_dim) — observed terminal latent window
            T_out: number of preceding frames to reconstruct
        Returns:
            (B, T_out, latent_dim)
        """
        B, W, D = z_window.shape

        enc = BidirectionalLSTM(
            hidden_size=self.hidden_dim, num_layers=self.num_layers,
            dropout_rate=self.dropout_rate if self.num_layers > 1 else 0.0)
        _, h_summary = enc(z_window, train=train)

        h_rep = jnp.tile(h_summary[:, None, :], (1, T_out, 1))
        pos = jnp.tile(jnp.linspace(0, 1, T_out)[None, :, None], (B, 1, 1))
        x = jnp.concatenate([h_rep, pos], axis=-1)
        x = nn.Dense(self.latent_dim, name="input_proj")(x)

        enc2 = BidirectionalLSTM(hidden_size=self.hidden_dim, num_layers=1, dropout_rate=0.0)
        enc2_out, _ = enc2(x, train=train)

        lstm_dec = nn.RNN(nn.LSTMCell(features=self.hidden_dim), name="lstm_decoder")
        dec_out = lstm_dec(enc2_out)

        flat = dec_out.reshape(B * T_out, self.hidden_dim)
        h = nn.Dense(self.hidden_dim * 2, name="mlp1")(flat)
        h = nn.LayerNorm(name="ln1")(h); h = nn.gelu(h)
        out = nn.Dense(self.latent_dim, name="output")(h)
        return out.reshape(B, T_out, self.latent_dim)


class BackwardFromTerminal(nn.Module):
    """Predict full preceding trajectory from a single terminal latent vector.

    Used for backward inference when only the terminal frame is observed
    (e.g., NDSI backward mode with static padding).
    """
    latent_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, z_T: jnp.ndarray, T_out: int, train: bool = True):
        """
        Args:
            z_T: (B, latent_dim) — terminal latent state
            T_out: number of preceding frames to reconstruct
        Returns:
            (B, T_out, latent_dim)
        """
        B = z_T.shape[0]

        z_rep = jnp.tile(z_T[:, None, :], (1, T_out, 1))
        pos = jnp.tile(jnp.linspace(0, 1, T_out)[None, :, None], (B, 1, 1))
        x = jnp.concatenate([z_rep, pos], axis=-1)
        x = nn.Dense(self.latent_dim, name="input_proj")(x)

        enc = BidirectionalLSTM(
            hidden_size=self.hidden_dim, num_layers=self.num_layers,
            dropout_rate=self.dropout_rate if self.num_layers > 1 else 0.0)
        enc_out, _ = enc(x, train=train)

        lstm_dec = nn.RNN(nn.LSTMCell(features=self.hidden_dim), name="lstm_decoder")
        dec_out = lstm_dec(enc_out)

        flat = dec_out.reshape(B * T_out, self.hidden_dim)
        h = nn.Dense(self.hidden_dim * 2, name="mlp1")(flat)
        h = nn.LayerNorm(name="ln1")(h); h = nn.gelu(h)
        out = nn.Dense(self.latent_dim, name="output")(h)
        return out.reshape(B, T_out, self.latent_dim)
