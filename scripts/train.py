import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

# isort: off
# Import the LeRobot data backend (PyAV native libs) before any openpi module that transitively
# imports OpenCV (cv2). Loading cv2 first triggers a native segfault during LeRobot data access on
# some systems (a cv2 / PyAV / Arrow native-library conflict). Keep data_loader first and do not let
# isort reorder these lines.
import openpi.training.data_loader as _data_loader

# torch.utils.tensorboard pulls in torch; keep it after data_loader so the LeRobot/PyAV backend is
# still the first native lib loaded (see the note above).
from torch.utils import tensorboard
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.summary as _summary
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# isort: on


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, dict[str, at.Array]],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        aux: dict[str, at.Array],
    ):
        # Returns (total_loss, metrics). The default implementation is mean(compute_loss(...)) with
        # empty metrics, so models without extra loss terms behave exactly as before.
        return model.compute_train_losses(rng, observation, actions, train=True, aux=aux, step=state.step)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, aux = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions, aux
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "lr": config.lr_schedule.create()(state.step),
        **loss_metrics,
    }
    # Per-module gradient norms. A head whose loss is flat is either starved of gradient (this is
    # ~0 => the loss does not reach it, or the LR schedule is not delivering), or getting gradient
    # and failing to use it (this is healthy but the loss still does not move => a capacity /
    # signal problem). Distinguishing those two from a loss curve alone is impossible.
    grads_by_module = grads.to_pure_dict()
    for module in ("phy_proj", "phy_action_proj", "phy_dist_head", "phy_proto_cls", "fvlmoe", "force_proj"):
        if (subtree := grads_by_module.get(module)) is not None:
            info[f"gnorm_{module}"] = optax.global_norm(subtree)
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path(jax_cache).expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # Persist per-log-interval metrics to a CSV in the checkpoint dir so the training curve is a
    # reproducible, auditable artifact. (tqdm_loggable swallows the per-step pbar.write output, so
    # scraping the console is unreliable.) Read it with examples/force/plot_training_loss.py.
    #
    # The key set is discovered from the first step's info rather than hard-coded, so every extra
    # loss a model reports (loss_flow / loss_dist / loss_proto / ...) lands in the CSV and in
    # TensorBoard without touching this script.
    metrics_path = config.checkpoint_dir / "metrics.csv"
    metrics_keys: list[str] | None = None

    # TensorBoard: view with `tensorboard --logdir checkpoints/<config>/<exp>/tensorboard`.
    tb_writer = tensorboard.SummaryWriter(str(config.checkpoint_dir / "tensorboard"))

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            for k, v in reduced_info.items():
                tb_writer.add_scalar(f"train/{k}", float(v), step)
            tb_writer.flush()
            if metrics_keys is None:
                metrics_keys = sorted(reduced_info)
                if not resuming or not metrics_path.exists():
                    metrics_path.write_text("step," + ",".join(metrics_keys) + "\n")
            with metrics_path.open("a") as metrics_file:
                metrics_file.write(f"{step}," + ",".join(f"{float(reduced_info[k]):.6f}" for k in metrics_keys) + "\n")
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    tb_writer.close()

    # Diagnostic summary: everything needed to work out WHY a loss did or did not move, readable
    # after the fact without the GPU or the checkpoint. See openpi/training/summary.py.
    if metrics_keys is not None:
        config_line = (
            f"{config.name} batch={config.batch_size} steps={config.num_train_steps} "
            f"peak_lr={getattr(config.lr_schedule, 'peak_lr', '?')} "
            f"warmup={getattr(config.lr_schedule, 'warmup_steps', '?')}"
        )
        summary_text = _summary.write_summary(
            metrics_path, config.checkpoint_dir / "training_summary.txt", config_line=config_line
        )
        print("\n" + summary_text)
        logging.info(f"Wrote training summary to {config.checkpoint_dir / 'training_summary.txt'}")

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
