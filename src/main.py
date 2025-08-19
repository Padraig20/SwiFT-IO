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
from wandb.sdk.wandb_run import Run
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

 

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_validation_epoch_end(self, trainer, pl_module):
        print(f"현재 epoch: {trainer.current_epoch}, best model path: {self.best_model_path}")
        super().on_validation_epoch_end(trainer, pl_module)

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        pass

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
    # — Slurm 재실행 시 사용할 W&B resume 옵션
    parser.add_argument("--resume", action="store_true",
                         help="If set, resume from an existing W&B run ID and load latest checkpoint")
    parser.add_argument("--run_id", type=str,
                         help="(When --resume) 이전에 사용하던 W&B run ID를 문자열로 전달합니다.")


    
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
    data_module.setup(stage='fit') # kimbo change
    
    

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

        # ①: resume 플래그가 켜진 경우, run_id=이전 W&B run ID, resume="allow" 설정
        if args.resume:
            wandb_logger = WandbLogger(
                project=args.project_name,
                name=args.experiment_name,
                id=args.run_id,                    # 반드시 이전 run_id를 넘겨야 합니다.
                resume="allow",                    # 기존 run을 이어붙이겠다는 의미
                config=vars(args),
                save_dir=args.default_root_dir,
                tags=tags, 
                log_model=False
            )

        else: 
            # ②: 새로 시작하는 경우, run_id=None, resume=None 설정
            wandb_logger = WandbLogger(
                project=args.project_name,
                name=args.experiment_name if hasattr(args, "experiment_name") else None,
                config=vars(args),
                save_dir=args.default_root_dir,  # 로그 저장 경로 설정
                tags=tags
            )
        
        # Lightning에서 사용할 logger 객체로 교체
        logger = wandb_logger
        # run_id를 args.id로 저장(추후에 Slurm 재실행 시 --> --run_id <args.id> 로 넣을 수 있게)
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
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        new_state_dict = OrderedDict()

        for k, v in ckpt.items():
            if 'model.' in k or '_forward_module.model.' in k:
                clean_key = k
                # 앞부분 prefix 제거
                for prefix in ['model.', '_forward_module.model.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                if not "head.weight" in k and not "head.bias" in k:
                    new_state_dict[clean_key] = v

        load_result = model.model.load_state_dict(new_state_dict, strict=False)
        print("=== Load State Dict Summary ===")
        if load_result.missing_keys:
            print("Missing keys (not loaded into model):")
            for k in load_result.missing_keys:
                print("  ", k)

        if load_result.unexpected_keys:
            print("Unexpected keys (in checkpoint but not in model):")
            for k in load_result.unexpected_keys:
                print("  ", k)

        
    # ------------ run -------------
    if args.test_only:
        trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path) # dataloaders=data_module
        return
    
    # → resume 플래그가 켜진 경우: W&B artifact에서 최신 checkpoint를 내려받아 이어서 학습
    if args.resume:
        # 1) W&B run 객체 가져오기 (resume="allow"로 이미 init함)
        run = wandb.init(project=args.project_name, id=args.run_id, resume="allow")

        # 2) Artifact에서 최신 체크포인트 다운로드
        artifact = run.use_artifact(f'{args.project_name}/best_model:latest')
        ckpt_dir = artifact.download()  

        # 3) 모델 구조 복원을 위한 args.json 또는 config.yaml 확인
        import json
        config_path = os.path.join(ckpt_dir, "config.json")  # or args.json, hparams.yaml
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                ckpt_config = json.load(f)
            print("[Resume] Loaded config from artifact:")
            for k, v in ckpt_config.items():
                if not hasattr(args, k):
                    setattr(args, k, v)
        else:
            print("[Resume Warning] No config.json found in artifact.")

        # 4) 로컬 .ckpt 파일 경로 추출
        import glob
        ckpt_list = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if len(ckpt_list) == 0:
            raise FileNotFoundError(f"Download 된 checkpoint 파일을 찾을 수 없습니다: {ckpt_dir}")
        latest_ckpt_path = ckpt_list[-1]  # 여러 개일 경우, 마지막(가장 최신) ckpt를 선택

        # 4) 모델·optimizer state를 자동으로 Lightning이 불러올 수 있도록 Trainer에 전달
        trainer.fit(model, datamodule=data_module, ckpt_path=latest_ckpt_path)
    else:
        # 새로운 run 또는 일반 재시작(resume_ckpt_path만 있는 경우)
        if args.resume_ckpt_path is None:
            trainer.fit(model, datamodule=data_module)
        else:
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

    # 학습 끝난 뒤에는 test (best ckpt 사용)
    trainer.test(model, dataloaders=data_module, ckpt_path="best")

    if args.save_encoder:
        model.save_encoder(args.save_encoder)

    # ✅ WandB 세션 종료
    if args.loggername == "wandb":
        wandb.finish()

if __name__ == "__main__":
    cli_main()