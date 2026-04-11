"""
shred_jax — JAX/Flax implementation of SHRED and LAPIS-SHRED.

SHRED: SHallow REcurrent Decoder for sparse-sensor spatiotemporal reconstruction.
LAPIS: LAtent Phase Inference from Short time sequences.
"""

# Core SHRED models
from .shred import (
    BidirectionalLSTM, MLPDecoder, FrameSHRED, Seq2SeqSHRED,
    weighted_mse_loss, create_active_weights, compute_metrics,
)

# Dataset classes
from .datasets import EnsembleSeq2SeqDataset, EnsembleFrameDataset

# Temporal dynamics models
from .temporal_models import ForwardFromWindow, BackwardFromWindow, BackwardFromTerminal

# Training
from .training import (
    train_ensemble_shred, train_ensemble_frame_shred,
    train_forward_model, train_backward_model,
)

# Inference
from .inference import (
    Seq2SeqSHREDEncoder, FlexibleSHREDDecoder,
    extract_encoder_params, extract_decoder_params,
    decode_latent_with_frozen_shred,
    extract_latent_trajectories_seq2seq, extract_latent_trajectories_frame,
    lapis_forward_inference_seq2seq, lapis_backward_inference_seq2seq,
    lapis_backward_inference_terminal_seq2seq, lapis_backward_inference_frame,
    shred_baseline_seq2seq, shred_baseline_frame,
)

# Utilities
from .utils import place_sensors, TeeLogger, to_json_safe
