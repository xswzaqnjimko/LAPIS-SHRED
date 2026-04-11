"""
Training loops for SHRED and LAPIS temporal models.

Provides:
  - SHRED training: train_ensemble_shred, train_ensemble_frame_shred
  - Temporal model training: train_forward_model, train_backward_model
  - Batch preparation: prepare_forward_batch, prepare_backward_batch,
      prepare_backward_terminal_batch
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from functools import partial

from .shred import weighted_mse_loss, create_active_weights


# SHRED training steps

@partial(jax.jit, static_argnums=(4,))
def train_step_shred(state, x, y, weights, hidden_size):
    def loss_fn(params):
        pred = state.apply_fn({"params": params}, x, train=True,
                              rngs={"dropout": jax.random.PRNGKey(0)})
        return weighted_mse_loss(pred, y, weights)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@partial(jax.jit)
def train_step_frame_shred(state, x, y, weights):
    def loss_fn(params):
        pred = state.apply_fn({"params": params}, x, train=True,
                              rngs={"dropout": jax.random.PRNGKey(0)})
        return weighted_mse_loss(pred, y, weights)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def train_ensemble_shred(model, dataset, active_mask, config, rng):
    """Train Seq2Seq SHRED on an ensemble of simulation sequences."""
    sample_x, _ = dataset[0]
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, sample_x[None, ...], train=False)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config.LR_SHRED, weight_decay=config.WEIGHT_DECAY))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx)
    weights_jax = jnp.array(create_active_weights(active_mask, config.ACTIVE_WEIGHT),
                            dtype=jnp.float32)

    print(f"  Training SHRED for {config.EPOCHS_SHRED} epochs ...")
    for epoch in range(config.EPOCHS_SHRED):
        rng, brng = jax.random.split(rng)
        batches = dataset.get_batches(config.BATCH_SIZE, shuffle=True, rng=brng)
        eloss = sum(float(train_step_shred(state, xb, yb, weights_jax,
                                           config.SEQ2SEQ_HIDDEN)[1])
                    for xb, yb in batches) if False else 0.0  # computed below
        eloss = 0.0
        for xb, yb in batches:
            state, loss = train_step_shred(state, xb, yb, weights_jax, config.SEQ2SEQ_HIDDEN)
            eloss += float(loss)
        eloss /= max(len(batches), 1)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{config.EPOCHS_SHRED}: loss = {eloss:.6f}")
    return state


def train_ensemble_frame_shred(model, dataset, active_mask, config, rng):
    """Train frame-by-frame SHRED on an ensemble of simulation grids."""
    sample_x = jnp.array(dataset.X_scaled[:1], dtype=jnp.float32)
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, sample_x, train=False)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config.LR_SHRED, weight_decay=config.WEIGHT_DECAY))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx)
    weights_jax = jnp.array(create_active_weights(active_mask, config.ACTIVE_WEIGHT),
                            dtype=jnp.float32)

    print(f"  Training Frame SHRED for {config.EPOCHS_SHRED} epochs "
          f"({len(dataset)} samples, lags={config.LAGS}) ...")
    for epoch in range(config.EPOCHS_SHRED):
        rng, brng = jax.random.split(rng)
        batches = dataset.get_batches(config.BATCH_SIZE, shuffle=True, rng=brng)
        eloss = 0.0
        for xb, yb in batches:
            state, loss = train_step_frame_shred(state, xb, yb, weights_jax)
            eloss += float(loss)
        eloss /= max(len(batches), 1)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{config.EPOCHS_SHRED}: loss = {eloss:.6f}")
    return state


# Forward model training

def prepare_forward_batch(latent_trajectories, T_originals, obs_len, max_T_future):
    n = len(latent_trajectories)
    D = latent_trajectories[0].shape[1]
    z_init_batch = np.zeros((n, obs_len, D), dtype=np.float32)
    z_target_batch = np.zeros((n, max_T_future, D), dtype=np.float32)
    mask_batch = np.zeros((n, max_T_future), dtype=np.float32)

    for i, (traj, T_orig) in enumerate(zip(latent_trajectories, T_originals)):
        traj = traj[:T_orig]
        W = min(obs_len, T_orig)
        z_init_batch[i, :W] = traj[:W]
        future = traj[obs_len:]
        T_fut = len(future)
        if T_fut > 0:
            z_target_batch[i, :T_fut] = future
            if T_fut < max_T_future:
                z_target_batch[i, T_fut:] = future[-1]
            mask_batch[i, :T_fut] = 1.0

    return jnp.array(z_init_batch), jnp.array(z_target_batch), jnp.array(mask_batch)


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def train_step_forward(state, z_init, z_target, mask, rng_key,
                       max_T_future, lambda_recon, lambda_anchor, lambda_shape):
    def loss_fn(params):
        z_pred = state.apply_fn({"params": params}, z_init, max_T_future,
                                train=True, rngs={"dropout": rng_key})
        sq = (z_pred - z_target) ** 2 * mask[:, :, None]
        l_recon = jnp.sum(sq) / (jnp.sum(mask) * z_pred.shape[-1] + 1e-8)
        l_anchor = jnp.mean((z_pred[:, 0, :] - z_init[:, -1, :]) ** 2)
        pd = z_pred[:, 1:, :] - z_pred[:, :-1, :]
        td = z_target[:, 1:, :] - z_target[:, :-1, :]
        md = mask[:, 1:] * mask[:, :-1]
        l_shape = jnp.sum((pd - td) ** 2 * md[:, :, None]) / (jnp.sum(md) * z_pred.shape[-1] + 1e-8)
        l_var = jnp.mean((jnp.var(z_pred, axis=1) - jnp.var(z_target, axis=1)) ** 2)
        total = lambda_recon * l_recon + lambda_anchor * l_anchor + lambda_shape * (l_shape + 0.5 * l_var)
        return total, (l_recon, l_anchor, l_shape, l_var)
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, aux


def train_forward_model(model, latent_trajectories, T_originals, obs_len, config, rng):
    D = latent_trajectories[0].shape[1]
    max_T_future = max(T - obs_len for T in T_originals)

    rng, irng = jax.random.split(rng)
    dummy_z = jnp.ones((1, obs_len, D))
    variables = model.init(irng, dummy_z, max_T_future, train=False)

    schedule = optax.cosine_decay_schedule(config.LR_FORWARD, config.EPOCHS_FORWARD, alpha=0.01)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.WEIGHT_DECAY))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx)

    z_init_b, z_target_b, mask_b = prepare_forward_batch(
        latent_trajectories, T_originals, obs_len, max_T_future)

    print(f"    obs_len={obs_len}, max_T_future={max_T_future}, D={D}")
    print(f"    z_init={z_init_b.shape}, z_target={z_target_b.shape}")

    best_loss, best_params = float("inf"), None
    lr = float(config.LAMBDA_RECON)
    la = float(config.LAMBDA_ANCHOR)
    ls = float(config.LAMBDA_SHAPE)

    for epoch in range(config.EPOCHS_FORWARD):
        rng, drng = jax.random.split(rng)
        state, loss, aux = train_step_forward(
            state, z_init_b, z_target_b, mask_b, drng,
            int(max_T_future), lr, la, ls)
        el = float(loss)
        if el < best_loss:
            best_loss = el
            best_params = jax.tree.map(lambda x: x.copy(), state.params)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            r, a, s, v = [float(x) for x in aux]
            print(f"    Epoch {epoch+1}/{config.EPOCHS_FORWARD}: loss={el:.6f} "
                  f"(recon={r:.6f} anchor={a:.6f} shape={s:.6f} var={v:.6f})")

    if best_params is not None:
        state = state.replace(params=best_params)
    print(f"    Best forward-model loss: {best_loss:.6f}")
    return state


# Backward model training (window-based, for KVS/KF experiments)

def prepare_backward_batch(latent_trajectories, T_originals, obs_len, max_T_prior):
    n = len(latent_trajectories)
    D = latent_trajectories[0].shape[1]
    z_window_batch = np.zeros((n, obs_len, D), dtype=np.float32)
    z_target_batch = np.zeros((n, max_T_prior, D), dtype=np.float32)
    z_init_batch = np.zeros((n, D), dtype=np.float32)
    mask_batch = np.zeros((n, max_T_prior), dtype=np.float32)

    for i, (traj, T_orig) in enumerate(zip(latent_trajectories, T_originals)):
        traj = traj[:T_orig]
        T_prior = T_orig - obs_len
        z_window_batch[i] = traj[-obs_len:]
        z_target_batch[i, :T_prior] = traj[:T_prior]
        if T_prior < max_T_prior:
            z_target_batch[i, T_prior:] = traj[T_prior - 1]
        z_init_batch[i] = traj[0]
        mask_batch[i, :T_prior] = 1.0

    return (jnp.array(z_window_batch), jnp.array(z_target_batch),
            jnp.array(z_init_batch), jnp.array(mask_batch))


@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def train_step_backward(state, z_window, z_target, z_init_batch, mask, rng_key,
                        max_T_prior, lambda_recon, lambda_anchor, lambda_shape):
    def loss_fn(params):
        z_pred = state.apply_fn({"params": params}, z_window, max_T_prior,
                                train=True, rngs={"dropout": rng_key})
        sq = (z_pred - z_target) ** 2 * mask[:, :, None]
        l_recon = jnp.sum(sq) / (jnp.sum(mask) * z_pred.shape[-1] + 1e-8)
        l_anchor = jnp.mean((z_pred[:, 0, :] - z_init_batch) ** 2)
        pd = z_pred[:, 1:, :] - z_pred[:, :-1, :]
        td = z_target[:, 1:, :] - z_target[:, :-1, :]
        md = mask[:, 1:] * mask[:, :-1]
        l_shape = jnp.sum((pd - td) ** 2 * md[:, :, None]) / (jnp.sum(md) * z_pred.shape[-1] + 1e-8)
        l_var = jnp.mean((jnp.var(z_pred, axis=1) - jnp.var(z_target, axis=1)) ** 2)
        total = lambda_recon * l_recon + lambda_anchor * l_anchor + lambda_shape * (l_shape + 0.5 * l_var)
        return total, (l_recon, l_anchor, l_shape, l_var)
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, aux


def train_backward_model(model, latent_trajectories, T_originals, obs_len, config, rng,
                         z_inits=None):
    """Train backward temporal model.

    Args:
        model: BackwardFromWindow or BackwardFromTerminal
        latent_trajectories: list of (T_i, D) latent arrays
        T_originals: list of original trajectory lengths
        obs_len: observation window length (for BackwardFromWindow) or ignored
        config: experiment config with LR_FORWARD, EPOCHS_FORWARD, etc.
        rng: JAX random key
        z_inits: optional list of initial latent states (for terminal-only backward)
    """
    D = latent_trajectories[0].shape[1]
    max_T = max(T_originals)

    # Detect model type by init signature
    is_terminal = _is_terminal_model(model, D, max_T, rng)

    if is_terminal:
        return _train_backward_terminal(model, latent_trajectories, T_originals,
                                        z_inits or [t[0] for t in latent_trajectories],
                                        config, rng)
    else:
        return _train_backward_window(model, latent_trajectories, T_originals,
                                      obs_len, config, rng)


def _is_terminal_model(model, D, max_T, rng):
    """Check if the model takes (B, D) input (terminal) vs (B, W, D) (window)."""
    try:
        rng_init = jax.random.split(rng)[0]
        model.init(rng_init, jnp.ones((1, D)), max_T - 1, train=False)
        return True
    except Exception:
        return False


def _train_backward_window(model, latent_trajectories, T_originals, obs_len, config, rng):
    """Train BackwardFromWindow model."""
    D = latent_trajectories[0].shape[1]
    max_T_prior = max(T - obs_len for T in T_originals)

    rng, irng = jax.random.split(rng)
    dummy = jnp.ones((1, obs_len, D))
    variables = model.init(irng, dummy, max_T_prior, train=False)

    schedule = optax.cosine_decay_schedule(config.LR_FORWARD, config.EPOCHS_FORWARD, alpha=0.01)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.WEIGHT_DECAY))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx)

    z_win_b, z_tgt_b, z_init_b, mask_b = prepare_backward_batch(
        latent_trajectories, T_originals, obs_len, max_T_prior)

    print(f"    z_window={z_win_b.shape}, z_target={z_tgt_b.shape}, max_T_prior={max_T_prior}")
    lr, la, ls = float(config.LAMBDA_RECON), float(config.LAMBDA_ANCHOR), float(config.LAMBDA_SHAPE)

    best_loss, best_params = float("inf"), None
    for epoch in range(config.EPOCHS_FORWARD):
        rng, drng = jax.random.split(rng)
        state, loss, aux = train_step_backward(
            state, z_win_b, z_tgt_b, z_init_b, mask_b, drng,
            int(max_T_prior), lr, la, ls)
        el = float(loss)
        if el < best_loss:
            best_loss = el
            best_params = jax.tree.map(lambda x: x.copy(), state.params)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            r, a, s, v = [float(x) for x in aux]
            print(f"    Epoch {epoch+1}/{config.EPOCHS_FORWARD}: loss={el:.6f} "
                  f"(recon={r:.6f} anchor={a:.6f} shape={s:.6f} var={v:.6f})")

    if best_params is not None:
        state = state.replace(params=best_params)
    print(f"    Best backward-model loss: {best_loss:.6f}")
    return state


# Backward model training (terminal-only, for NDSI backward)

def prepare_backward_terminal_batch(latent_trajectories, T_originals, z_inits, max_T_out):
    n = len(latent_trajectories)
    D = latent_trajectories[0].shape[1]
    z_T_batch = np.zeros((n, D), dtype=np.float32)
    z_target_batch = np.zeros((n, max_T_out, D), dtype=np.float32)
    z_init_batch = np.zeros((n, D), dtype=np.float32)
    mask_batch = np.zeros((n, max_T_out), dtype=np.float32)

    for i, (traj, T_orig, z_init) in enumerate(zip(latent_trajectories, T_originals, z_inits)):
        traj_orig = traj[:T_orig]
        T_out = T_orig - 1
        z_T_batch[i] = traj_orig[-1]
        z_target_batch[i, :T_out] = traj_orig[:-1]
        if T_out < max_T_out:
            z_target_batch[i, T_out:] = traj_orig[-2]
        z_init_batch[i] = z_init
        mask_batch[i, :T_out] = 1.0

    return (jnp.array(z_T_batch), jnp.array(z_target_batch),
            jnp.array(z_init_batch), jnp.array(mask_batch))


@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def train_step_backward_terminal(state, z_T_batch, z_target_batch, z_init_batch,
                                 mask_batch, rng_key,
                                 max_T_out, lambda_recon, lambda_conv, lambda_shape):
    def loss_fn(params):
        z_pred = state.apply_fn({"params": params}, z_T_batch, max_T_out,
                                train=True, rngs={"dropout": rng_key})
        sq = (z_pred - z_target_batch) ** 2 * mask_batch[:, :, None]
        l_recon = jnp.sum(sq) / (jnp.sum(mask_batch) * z_pred.shape[-1] + 1e-8)
        l_conv = jnp.mean((z_pred[:, 0, :] - z_init_batch) ** 2)
        pd = z_pred[:, 1:, :] - z_pred[:, :-1, :]
        td = z_target_batch[:, 1:, :] - z_target_batch[:, :-1, :]
        md = mask_batch[:, 1:] * mask_batch[:, :-1]
        l_shape = jnp.sum((pd - td) ** 2 * md[:, :, None]) / (jnp.sum(md) * z_pred.shape[-1] + 1e-8)
        l_var = jnp.mean((jnp.var(z_pred, axis=1) - jnp.var(z_target_batch, axis=1)) ** 2)
        total = lambda_recon * l_recon + lambda_conv * l_conv + lambda_shape * (l_shape + 0.5 * l_var)
        return total, (l_recon, l_conv, l_shape, l_var)
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, aux


def _train_backward_terminal(model, latent_trajectories, T_originals, z_inits, config, rng):
    """Train BackwardFromTerminal model."""
    print("  Training BackwardFromTerminal ...")
    D = latent_trajectories[0].shape[1]
    max_T_out = max(T_originals) - 1

    rng, irng = jax.random.split(rng)
    dummy = jnp.ones((1, D))
    variables = model.init(irng, dummy, max_T_out, train=False)

    schedule = optax.cosine_decay_schedule(config.LR_FORWARD, config.EPOCHS_FORWARD, alpha=0.01)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.WEIGHT_DECAY))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx)

    z_T_b, z_tgt_b, z_init_b, mask_b = prepare_backward_terminal_batch(
        latent_trajectories, T_originals, z_inits, max_T_out)

    print(f"    z_T={z_T_b.shape}, z_target={z_tgt_b.shape}, max_T_out={max_T_out}")
    lr, lc, ls = float(config.LAMBDA_RECON), float(config.LAMBDA_ANCHOR), float(config.LAMBDA_SHAPE)

    best_loss, best_params = float("inf"), None
    for epoch in range(config.EPOCHS_FORWARD):
        rng, drng = jax.random.split(rng)
        state, loss, aux = train_step_backward_terminal(
            state, z_T_b, z_tgt_b, z_init_b, mask_b, drng,
            int(max_T_out), lr, lc, ls)
        el = float(loss)
        if el < best_loss:
            best_loss = el
            best_params = jax.tree.map(lambda x: x.copy(), state.params)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            r, c, s, v = [float(x) for x in aux]
            print(f"    Epoch {epoch+1}/{config.EPOCHS_FORWARD}: loss={el:.6f} "
                  f"(recon={r:.6f} conv={c:.6f} shape={s:.6f} var={v:.6f})")

    if best_params is not None:
        state = state.replace(params=best_params)
    print(f"    Best backward-model loss: {best_loss:.6f}")
    return state
