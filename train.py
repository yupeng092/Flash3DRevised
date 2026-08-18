import os
import sys
import time
import logging
import torch
import hydra
import torch.optim as optim

# Ensure the project root is first on sys.path so the local ``datasets/``
# package takes precedence over any HuggingFace ``datasets`` wheel installed in
# the environment.  Without this, ``from datasets.util import ...`` resolves to
# the HF package (which has no ``util`` submodule) and crashes at import time.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path or sys.path[0] != _PROJECT_ROOT:
    sys.path.insert(0, _PROJECT_ROOT)

from ema_pytorch import EMA
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import seed_everything
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy

from evaluation.evaluator import Evaluator
from datasets.util import create_datasets
from trainer import Trainer


def run_epoch(fabric,
              trainer,
              ema,
              train_loader,
              val_loader,
              optimiser,
              lr_scheduler,
              evaluator):
    """Run a single epoch of training and validation
    """
    cfg = trainer.cfg
    trainer.model.set_train()

    if fabric.is_global_zero:
        logging.info("Training on epoch {}".format(trainer.epoch))

    for batch_idx, inputs in enumerate(train_loader):
        # instruct the model which novel frames to render
        inputs["target_frame_ids"] = cfg.model.gauss_novel_frames
        losses, outputs = trainer(inputs)

        optimiser.zero_grad(set_to_none=True)
        fabric.backward(losses["loss/total"])
        optimiser.step()
        if ema is not None:
            ema.update()
        
        step = trainer.step

        early_phase = batch_idx % trainer.cfg.run.log_frequency == 0 and step < 6000
        if fabric.is_global_zero:
            learning_rate = lr_scheduler.get_lr()
            if type(learning_rate) is list:
                learning_rate = max(learning_rate)
            # Always print the step loss to stdout so CPU smoke runs can observe
            # loss trajectory without the optional Neptune logger.
            total = float(losses["loss/total"].detach().cpu())
            print(f"[epoch {trainer.epoch} | step {step:>5} | batch {batch_idx:>3}] "
                  f"loss/total={total:.6f} lr={learning_rate:.2e}", flush=True)
            # save the loss and scales
            trainer.log_scalars("train", outputs, losses, learning_rate)

            # log less frequently after the first 2000 steps to save time & disk space
            late_phase = step % 2000 == 0
            # save the visual results
            if early_phase or late_phase:
                trainer.log("train", inputs, outputs)
            # save the model
            if step % cfg.run.save_frequency == 0 and step != 0:
                trainer.model.save_model(optimiser, step, ema)
            # save the validation results
            early_phase = (step < 6000) and (step % 500 == 0)
            if (early_phase or step % cfg.run.val_frequency == 0): # and step != 0:
                model_eval = ema if ema is not None else trainer.model
                trainer.validate(model_eval, evaluator, val_loader, device=fabric.device)

        if (early_phase or step % cfg.run.val_frequency == 0): # and step != 0:
            if fabric.device.type == "cuda":
                torch.cuda.empty_cache()
            elif fabric.device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.empty_cache()
            
        trainer.step += 1
        lr_scheduler.step()

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    # set up the output directory
    try:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg['runtime']['output_dir']
    except Exception:
        # Fallback for non-@hydra.main entry points (e.g. scripts/train_cpu.py
        # which composes the config directly to bypass a Python 3.14 argparse
        # incompatibility in Hydra 1.3.5).
        output_dir = cfg.get("run", {}).get("dirpath", None) or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "outputs", "cpu_train_run"
        )
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    logging.info(f"Working dir: {output_dir}")
    # set up random set
    torch.set_float32_matmul_precision('high')
    seed_everything(cfg.run.random_seed)
    accelerator = getattr(cfg.train, "accelerator", "cuda")
    if accelerator == "npu":
        try:
            # Importing torch_npu registers the ``npu`` device and Lightning
            # accelerator before Fabric is constructed.
            import torch_npu  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "NPU training requires torch_npu. Install the wheel matched to CANN."
            ) from error
    # set up training precision
    # A single-device CPU smoke test should not initialise DDP.  Keeping DDP
    # for multi-device runs preserves the original distributed training path.
    devices = cfg.train.num_gpus
    strategy = DDPStrategy(find_unused_parameters=True) if devices > 1 else "auto"
    fabric = Fabric(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=cfg.train.mixed_precision
    )
    fabric.launch()
    fabric.barrier()
    print("Loaded datasets")
    # set up model
    trainer = Trainer(cfg)
    model = trainer.model
    # set up optimiser
    optimiser = optim.Adam(model.parameters_to_train, cfg.optimiser.learning_rate)
    def lr_lambda(*args):
        threshold = cfg.optimiser.scheduler_lambda_step_size
        if trainer.step < threshold:
            return 1.0
        else:
            return 0.1
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda
    )
    if cfg.train.ema.use and fabric.is_global_zero:
        ema = EMA(  
            model, 
            beta=cfg.train.ema.beta,
            update_every=cfg.train.ema.update_every,
            update_after_step=cfg.train.ema.update_after_step
        )
        ema = fabric.to_device(ema)
    else:
        ema = None
    # set up checkpointing
    if (ckpt_dir := model.checkpoint_dir()).exists():
        # resume training
        model.load_model(ckpt_dir, optimiser=optimiser)
    elif cfg.train.load_weights_folder:
        model.load_model(cfg.train.load_weights_folder)
    trainer, optimiser = fabric.setup(trainer, optimiser)
    # set up dataset
    train_dataset, train_loader = create_datasets(cfg, split="train")
    train_loader = fabric.setup_dataloaders(train_loader)
    if fabric.is_global_zero:
        if cfg.train.logging:
            # Neptune credentials are optional for local/CPU smoke tests.  Keep
            # the import lazy so a checkout without the private token module
            # can still train with ``train.logging=false``.
            from misc.logger import setup_logger
            trainer.set_logger(setup_logger(cfg))
        val_dataset, val_loader = create_datasets(cfg, split="val")
        evaluator = Evaluator()
        evaluator = fabric.to_device(evaluator)
    else:
        val_loader = None
        evaluator = None
    # launch training
    trainer.epoch = 0
    trainer.start_time = time.time()
    for trainer.epoch in range(cfg.optimiser.num_epochs):
        run_epoch(fabric, trainer, ema, train_loader, val_loader, optimiser, lr_scheduler, evaluator)


if __name__ == "__main__":
    main()
