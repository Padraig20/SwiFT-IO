"""
Training script for LSTM Baseline

LSTM baseline for fMRI emotion decoding
Provides fair comparison with SwiFT-IO by using same data, tasks, and evaluation

Usage:
    python train_lstm_baseline.py \\
        --model lstm_encoder \\
        --decoder lstm_regression_head \\
        --lstm_hidden_dim 256 \\
        --lstm_num_layers 2 \\
        --learning_rate 1e-4

Author: For LSTM baseline comparison
Date: 2025-10-14
"""

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from module.utils.data_module import fMRIDataModule
from module.pl_classifier import LitClassifier
import wandb


class CustomModelCheckpoint(ModelCheckpoint):
    """Custom checkpoint callback with Wand B artifact logging"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_validation_epoch_end(self, trainer, pl_module):
        print(f"Current epoch: {trainer.current_epoch}, best model path: {self.best_model_path}")
        super().on_validation_epoch_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_path = self.best_model_path
        print(f"Checkpoint path: {checkpoint_path}")

        # Best performance metric
        best_metric = trainer.callback_metrics.get('valid_acc') if 'valid_acc' in trainer.callback_metrics else trainer.callback_metrics.get('valid_mse')

        artifact = wandb.Artifact('best_lstm_model', type='model')

        if os.path.isfile(checkpoint_path):
            artifact.add_file(checkpoint_path)
            artifact.metadata = {
                'valid_metric': float(best_metric) if best_metric is not None else None,
                'epoch': trainer.current_epoch
            }
            wandb.log_artifact(artifact)
        else:
            print(f"Checkpoint path is not a valid file: {checkpoint_path}")

        return super().on_save_checkpoint(trainer, pl_module, checkpoint)


def cli_main():
    # ------------ args -------------
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)

    # Basic args
    parser.add_argument("--seed", default=777, type=int, help="Random seed")
    parser.add_argument("--dataset_name", type=str, default="HBN", help="Dataset name")
    parser.add_argument("--downstream_task", type=str, default="emotions", help="Downstream task")
    parser.add_argument("--downstream_task_type", type=str, default="regression", help="Task type: classification or regression")
    parser.add_argument("--loggername", default="wandb", type=str, help="Logger name")
    parser.add_argument("--project_name", default="moviefmri", type=str, help="Project name")
    parser.add_argument("--experiment_name", default="lstm_emotions", type=str, help="Experiment name")

    # Checkpoint args
    parser.add_argument("--resume_ckpt_path", type=str, help="Path to resume checkpoint")
    parser.add_argument("--test_only", action='store_true', help="Test only mode")
    parser.add_argument("--test_ckpt_path", type=str, help="Path to test checkpoint")
    parser.add_argument("--valid_only", action='store_true', help="Validation only (no test)")

    # LSTM-specific args
    parser.add_argument("--lstm_hidden_dim", type=int, default=256, help="LSTM hidden dimension")
    parser.add_argument("--lstm_num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--lstm_dropout", type=float, default=0.3, help="LSTM dropout rate")
    parser.add_argument("--lstm_bidirectional", action='store_true', help="Use bidirectional LSTM")
    parser.add_argument("--lstm_pooling", type=str, default='adaptive', choices=['adaptive', 'avgpool', 'flatten'], help="Spatial pooling method")
    parser.add_argument("--lstm_pooled_dim", type=int, default=16, help="Pooled spatial dimension")
    parser.add_argument("--lstm_decoder_hidden", type=int, default=128, help="Decoder hidden dimension")

    temp_args, _ = parser.parse_known_args()

    # Set classifier and dataset
    Classifier = LitClassifier
    Dataset = fMRIDataModule

    # Add model and data specific args
    parser = Classifier.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    # Set model and decoder defaults for LSTM after all args are added
    parser.set_defaults(model="lstm_encoder", decoder="lstm_regression_head")

    _, _ = parser.parse_known_args()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Override parameters
    max_epochs = args.max_epochs
    num_nodes = args.num_nodes
    devices = args.devices
    project_name = args.project_name
    image_path = args.image_path

    if temp_args.resume_ckpt_path is not None:
        # Resume previous experiment
        from module.utils.neptune_utils import get_prev_args
        args = get_prev_args(args.resume_ckpt_path, args)
        exp_id = args.id
        args.project_name = project_name
        args.max_epochs = max_epochs
        args.num_nodes = num_nodes
        args.devices = devices
        args.image_path = image_path
    else:
        exp_id = None

    setattr(args, "default_root_dir", f"output/{args.project_name}")

    # ------------ EarlyStopping -------------
    if args.downstream_task_type == "classification":
        early_stop_callback = EarlyStopping(
            monitor='valid_acc',
            patience=10,
            verbose=True,
            mode='max',
            check_on_train_epoch_end=True
        )
    else:  # regression
        early_stop_callback = EarlyStopping(
            monitor='valid_mse',
            patience=10,
            verbose=True,
            mode='min',
            check_on_train_epoch_end=True
        )

    # ------------ data -------------
    data_module = Dataset(**vars(args))
    pl.seed_everything(args.seed)
    data_module.setup(stage='fit')

    # ------------ logger -------------
    if args.loggername == "wandb":
        # Only initialize WandB on global rank 0 to avoid multi-process conflicts
        import torch.distributed as dist
        is_global_zero = not dist.is_initialized() or dist.get_rank() == 0

        if is_global_zero:
            API_KEY = os.environ.get("WANDB_API_KEY")
            wandb.login(key=API_KEY)

        tags = ["lstm_baseline"]
        if args.experiment_name is not None:
            tags.append(args.experiment_name)
        if args.valid_only:
            tags.append("valid_only")
        if args.test_only:
            tags.append("test_only")

        os.environ["WANDB_ANONYMOUS"] = "allow"
        logger = WandbLogger(
            project=args.project_name,
            name=args.experiment_name if hasattr(args, "experiment_name") else None,
            config=vars(args),
            save_dir=args.default_root_dir,
            tags=tags,
            anonymous="allow"
        )

        if exp_id is None:
            setattr(args, "id", logger.experiment.id)

        print(f"default_root_dir: {args.default_root_dir}")
        dirpath = os.path.join(args.default_root_dir, str(logger.version) if logger.version is not None else "default_version")
    else:
        raise Exception("Only wandb logger is supported for LSTM baseline")

    # ------------ callbacks -------------
    if args.downstream_task_type == "classification":
        checkpoint_callback = CustomModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_acc",
            filename="lstm-{epoch:02d}-{valid_acc:.2f}",
            save_last=True,
            mode="max",
        )
    else:  # regression
        checkpoint_callback = CustomModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_mse",
            filename="lstm-{epoch:02d}-{valid_mse:.4f}",
            save_last=True,
            mode="min",
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]

    # ------------ trainer -------------
    print("Training LSTM Baseline")
    print(f"  Model: {args.model}")
    print(f"  Decoder: {args.decoder}")
    print(f"  LSTM hidden dim: {args.lstm_hidden_dim}")
    print(f"  LSTM layers: {args.lstm_num_layers}")
    print(f"  Spatial pooling: {args.lstm_pooling} ({args.lstm_pooled_dim}^3)")
    print(f"  Learning rate: {args.learning_rate}")

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )

    # ------------ model -------------
    model = Classifier(data_module=data_module, **vars(args))

    # ------------ run -------------
    if args.test_only:
        trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path)
    else:
        if args.resume_ckpt_path is None:
            # New run
            trainer.fit(model, datamodule=data_module)
        else:
            # Resume existing run
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

        trainer.test(model, dataloaders=data_module, ckpt_path="best")

    # Finish wandb session
    if args.loggername == "wandb":
        wandb.finish()

    print("\nLSTM Baseline Training Complete!")
    print(f"Results saved to: {dirpath}")


if __name__ == "__main__":
    cli_main()
