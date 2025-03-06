import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# from module import LitClassifier
import neptune.new as neptune
from module.utils.data_module import fMRIDataModule
from module.pl_classifier import LitClassifier
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger


import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_validation_epoch_end(self, trainer, pl_module):
        # 검증 종료 후에 실행되는 부분
        print(f"현재 epoch: {trainer.current_epoch}, best model path: {self.best_model_path}")
        super().on_validation_epoch_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_path = self.best_model_path  # 최고 성능 체크포인트 경로
        print(f"Checkpoint path: {checkpoint_path}")
        artifact = wandb.Artifact('best_model', type='model')
        if os.path.isfile(checkpoint_path):
            artifact.add_file(checkpoint_path)  # 체크포인트 파일 추가
            wandb.log_artifact(artifact)  # 아티팩트로 로깅
        else:
            print(f"Checkpoint path is not a valid file: {checkpoint_path}")
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)
    
    
def cli_main():

    # ------------ args -------------
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=777, type=int, help="random seeds. recommend aligning this argument with data split number to control randomness")
    parser.add_argument("--dataset_name", type=str, default="UKB")
    parser.add_argument("--downstream_task", type=str, default="emotion", help="downstream task")
    parser.add_argument("--downstream_task_type", type=str, default="default", help="select either classification or regression according to your downstream task")
    parser.add_argument("--loggername", default="default", type=str, help="A name of logger")
    parser.add_argument("--project_name", default="default", type=str, help="A name of project (Neptune)")
    parser.add_argument("--resume_ckpt_path", type=str, help="A path to previous checkpoint. Use when you want to continue the training from the previous checkpoints")
    parser.add_argument("--load_model_path", type=str, help="A path to the pre-trained model weight file (.pth)")
    parser.add_argument("--test_only", action='store_true', help="specify when you want to test the checkpoints (model weights)")
    parser.add_argument("--test_ckpt_path", type=str, help="A path to the previous checkpoint that intends to evaluate (--test_only should be True)")
    parser.add_argument("--freeze_feature_extractor", action='store_true', help="Whether to freeze the feature extractor (for evaluating the pre-trained weight)")
    parser.add_argument("--grad_clip", action='store_true', help="whether to use scheduler")
    
    parser.add_argument("--save_encoder", type=str, default=None, help="Path to save the SwiFT encoder after training, if wanted")
    parser.add_argument("--experiment_name", default=None, type=str, help="A name of the experiment (WandB)") # kimbo change
    parser.add_argument("--valid_only", action='store_true', help="disable running _evaluate_metrics(mode='test') at validation stage") # kimbo change
    parser.set_defaults(valid_only=False)  # kimbo change
    # valid only > pl.classifier or dataloader에서 둘 다 불러올 수 있음. > self.trainer.hparams.
    
    temp_args, _ = parser.parse_known_args()

    # Set classifier
    Classifier = LitClassifier
    
    # Set dataset
    Dataset = fMRIDataModule

    
    # add two additional arguments
    parser = Classifier.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    #override parameters
    max_epochs = args.max_epochs
    num_nodes = args.num_nodes
    devices = args.devices
    project_name = args.project_name
    image_path = args.image_path

    if temp_args.resume_ckpt_path is not None:
        # resume previous experiment
        from module.utils.neptune_utils import get_prev_args
        args = get_prev_args(args.resume_ckpt_path, args)
        exp_id = args.id
        # override max_epochs if you hope to prolong the training
        args.project_name = project_name
        args.max_epochs = max_epochs
        args.num_nodes = num_nodes
        args.devices = devices
        args.image_path = image_path       
    else:
        exp_id = None
    
    setattr(args, "default_root_dir", f"output/{args.project_name}")

    
    # ------------ EarlyStopping 설정 -------------
    if args.downstream_task_type == "classification":
        early_stop_callback = EarlyStopping(
            monitor='valid_acc',        # 모니터할 메트릭 (예: valid_loss, valid_acc)
            patience=20,                  # 성능 향상이 없을 경우 학습을 멈출 때까지 기다릴 에포크 수
            verbose=True,                # 진행 상태 출력 여부
            mode='max',                  # 'min' (loss가 낮을수록 좋음) 또는 'max' (accuracy가 높을수록 좋음)
            check_on_train_epoch_end=True  # train epoch가 끝날 때마다 체크
        )

    if args.downstream_task_type == "regression":
        early_stop_callback = EarlyStopping(
            monitor='valid_mse',        # 모니터할 메트릭 (예: valid_loss, valid_acc)
            patience=20,                  # 성능 향상이 없을 경우 학습을 멈출 때까지 기다릴 에포크 수
            verbose=True,                # 진행 상태 출력 여부
            mode='min',                  # 'min' (loss가 낮을수록 좋음) 또는 'max' (accuracy가 높을수록 좋음)
            check_on_train_epoch_end=True  # train epoch가 끝날 때마다 체크
        )

    # ------------ data -------------
    data_module = Dataset(**vars(args))
    pl.seed_everything(args.seed)
    
    

    # ------------ logger -------------
    # log_every_n_steps = int(data_module.test_loader.dataset.total_len / (args.batch_size * int(num_nodes) * int(devices))) - 1
    # log_every_n_steps = 1 if log_every_n_steps == 0 else log_every_n_steps
    # log_every_n_steps = 50 if log_every_n_steps > 50 else log_every_n_steps
    # print("log_every_n_steps:",log_every_n_steps)


    if args.loggername == "tensorboard":
        # logger = True  # tensor board is a default logger of Trainer class
        dirpath = args.default_root_dir
        logger = TensorBoardLogger(dirpath)
    elif args.loggername == "neptune":
        API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
        # project_name should be "WORKSPACE_NAME/PROJECT_NAME"
        run = neptune.init(api_token=API_KEY, project=args.project_name, capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False, run=exp_id)
        
        if exp_id == None:
            setattr(args, "id", run.fetch()['sys']['id'])

        logger = NeptuneLogger(run=run, log_model_checkpoints=False)
        dirpath = os.path.join(args.default_root_dir, logger.version)

    elif args.loggername == "wandb":
        API_KEY = os.environ.get("WANDB_API_KEY")
        wandb.login(key=API_KEY)  # W&B 로그인 (생략 가능)

        tags = [] # kimbo change
        if args.experiment_name is not None:
            tags.append(args.experiment_name)
        if args.valid_only:
            tags.append("valid_only")
        if args.test_only:
            tags.append("test_only")  # kimbo change
        # run의 기타 특징들을 태그로 추가 가능
        # tags.extend(["in_production", "preemptible", "baseline"])

        # W&B Logger 설정
        logger = WandbLogger(
            project=args.project_name,
            name=args.experiment_name if hasattr(args, "experiment_name") else None,
            config=vars(args),
            save_dir=args.default_root_dir,  # 로그 저장 경로 설정
            tags = tags
        )

        if exp_id is None:
            setattr(args, "id", logger.experiment.id)  # W&B Experiment ID 저장
        print(f"default_root_dir: {args.default_root_dir}") # output/moviefmri
        # dirpath = os.path.join(args.default_root_dir, logger.version)
        dirpath = os.path.join(args.default_root_dir, str(logger.version) if logger.version is not None else "default_version")


    else:
        raise Exception("Wrong logger name.")

    # ------------ callbacks -------------
    # callback for classification task
    if args.downstream_task_type == "classification":
        checkpoint_callback = CustomModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_acc",
            filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
            save_last=True,
            mode="max",
        )
    # callback for regression task
    else:
        checkpoint_callback = CustomModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_mse",
            filename="checkpt-{epoch:02d}-{valid_mse:.2f}",
            save_last=True,
            mode="min",
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor, early_stop_callback] # kimbo change

    # ------------ trainer -------------
    if args.grad_clip:
        print('using gradient clipping')
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
            track_grad_norm=-1
        )
    else:
        print('not using gradient clipping')
        print(args)
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            check_val_every_n_epoch=1,
            #val_check_interval=100 if not args.scalability_check else None,
            callbacks=callbacks,
        )

    # ------------ model -------------
    model = Classifier(data_module = data_module, **vars(args))  # swifun: Classifier(**vars(args)) 

    if args.load_model_path is not None:
        print(f'loading model from {args.load_model_path}')
        path = args.load_model_path
        ckpt = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if 'model.' in k: #transformer-related layers
                if not "head.weight" in k and not "head.bias" in k:
                    new_state_dict[k.removeprefix("model.")] = v
        model.model.swinViT.load_state_dict(new_state_dict)

    # ------------ run -------------
    if args.test_only:
        trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path) # dataloaders=data_module
    else:
        if args.resume_ckpt_path is None:
            # New run
            trainer.fit(model, datamodule=data_module)
        else:
            # Resume existing run
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

        trainer.test(model, dataloaders=data_module) # 여기서 Best ckpt를 가져와야함. 
    
    if args.save_encoder:
        model.save_encoder(args.save_encoder)

    # ✅ WandB 세션 종료
    if args.loggername == "wandb":
        wandb.finish()



if __name__ == "__main__":
    cli_main()