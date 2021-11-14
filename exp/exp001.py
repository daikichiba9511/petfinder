""" exp000

Ref
* https://www.kaggle.com/phalanx/train-swin-t-pytorch-lightning
"""

import os
import warnings
from glob import glob
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from box import Box
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_lightning import LightningDataModule, LightningModule, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import StratifiedKFold
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from timm import create_model
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm import tqdm

sns.set()
warnings.filterwarnings("ignore")

config = {
    "expname": os.path.basename(__file__).split(".")[0],
    "train": True,
    "train_epoch": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # "train_epoch": [0],
    "inference": True,
    "device": "cuda",
    "seed": 2021,
    "root": "/content/input/petfinder-pawpularity-score/",
    "n_splits": 10,
    "epoch": 20,
    "trainer": {
        "gpus": 1,
        "accumulate_grad_batches": 16,
        "progress_bar_refresh_rate": 1,
        "fast_dev_run": False,
        "num_sanity_val_steps": 0,
        "resume_from_checkpoint": None,
        # "precision": 16,
    },
    "transform": {"name": "get_default_transforms", "image_size": 384},
    "train_loader": {
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": False,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 4,
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": False,
        "drop_last": False,
    },
    "test_loader": {
        "batch_size": 4,
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": False,
        "drop_last": False,
    },
    "model": {"name": "swin_large_patch4_window12_384_in22k", "output_dim": 1},
    "optimizer": {
        "name": "optim.AdamW",
        "params": {"lr": 1e-5},
    },
    "scheduler": {
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params": {
            "T_0": 20,
            "eta_min": 1e-4,
        },
    },
    "loss": "nn.BCEWithLogitsLoss",
}

config = Box(config)
pprint(config)

torch.autograd.set_detect_anomaly(True)
seed_everything(config.seed)


#######################
# dataset
#######################
class PetfinderDataset(Dataset):
    def __init__(self, df, image_size=224):
        self._X = df["Id"].values
        self._y = None
        if "Pawpularity" in df.keys():
            self._y = df["Pawpularity"].values
        self._transform = T.Resize([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image


class PetfinderDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            PetfinderDataset(self._train_df, self._cfg.transform.image_size)
            if train
            else PetfinderDataset(self._val_df, self._cfg.transform.image_size)
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)


#######################
# Augmentation
#######################
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB


def get_default_transforms():
    transform = {
        "train": T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        "val": T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        "test": T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    }
    return transform


#######################
# Model
#######################
def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()
        self.transform = get_default_transforms()
        self.save_hyperparameters(cfg)

    def __build_model(self):
        self.backbone = create_model(
            self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.output_dim)
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return out

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "train")
        return {"loss": loss, "pred": pred, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "val")
        return {"pred": pred, "labels": labels}

    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float() / 100.0
        images = self.transform[mode](images)

        if torch.rand(1)[0] < 0.5 and mode == "train":
            mix_images, target_a, target_b, lam = mixup(images, labels, alpha=0.5)
            logits = self.forward(mix_images).squeeze(1)
            loss = self._criterion(logits, target_a) * lam + (
                1 - lam
            ) * self._criterion(logits, target_b)
        else:
            logits = self.forward(images).squeeze(1)
            loss = self._criterion(logits, labels)

        pred = logits.sigmoid().detach().cpu() * 100.0
        labels = labels.detach().cpu() * 100.0
        return loss, pred, labels

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "val")

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out["pred"], out["labels"]
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = torch.sqrt(((labels - preds) ** 2).mean())
        self.log(f"{mode}_loss", metrics)

    def check_gradcam(
        self, dataloader, target_layer, target_category, reshape_transform=None
    ):
        cam = GradCAMPlusPlus(
            model=self,
            target_layers=[target_layer],
            use_cuda=self.cfg.trainer.gpus,
            reshape_transform=reshape_transform,
        )

        org_images, labels = iter(dataloader).next()
        cam.batch_size = len(org_images)
        images = self.transform["val"](org_images)
        images = images.to(self.device)
        logits = self.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        labels = labels.cpu().numpy()

        grayscale_cam = cam(
            input_tensor=images, target_category=target_category, eigen_smooth=True
        )
        org_images = org_images.detach().cpu().numpy().transpose(0, 2, 3, 1) / 255.0
        return org_images, grayscale_cam, pred, labels

    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        scheduler = eval(self.cfg.scheduler.name)(
            optimizer, **self.cfg.scheduler.params
        )
        return [optimizer], [scheduler]


def train(config):
    df = pd.read_csv(os.path.join(config.root, "train.csv"))
    df["Id"] = df["Id"].apply(lambda x: os.path.join(config.root, "train", x + ".jpg"))
    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        if fold not in config.train_epoch:
            continue
        print("#" * 8 + f"  Fold: {fold}  " + "#" * 8)
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        datamodule = PetfinderDataModule(train_df, val_df, config)
        model = Model(config)
        earystopping = EarlyStopping(monitor="val_loss")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            dirpath=f"./output/{config.model.name}",
            filename=f"best_loss_{fold}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
        )
        logger = TensorBoardLogger(
            save_dir="./output/tb_logs", name=f"{config.model.name}"
        )

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.epoch,
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **config.trainer,
        )
        trainer.fit(model, datamodule=datamodule)

        # analysis of model
        # class_activation_map(train_df, val_df, fold)
        visualize_result(config, fold)

        torch.cuda.empty_cache()


#######################
# Class Activation Map
#######################
def reshape_transform(tensor, height=12, width=12):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # like in CNNs.
    result = result.permute(0, 3, 1, 2)
    return result


def class_activation_map(train_df, val_df, fold: int):
    model = Model(config).to(config.device).eval()
    model.load_state_dict(
        torch.load(f"./output/{config.model.name}/best_loss_{fold}.ckpt")["state_dict"]
    )
    config.val_loader.batch_size = 16
    datamodule = PetfinderDataModule(train_df, val_df, config)
    images, grayscale_cams, preds, labels = model.check_gradcam(
        datamodule.val_dataloader(),
        target_layer=model.backbone.layers[-1].blocks[-1].norm1,
        target_category=None,
        reshape_transform=reshape_transform,
    )
    plt.figure(figsize=(12, 12))
    for it, (image, grayscale_cam, pred, label) in enumerate(
        zip(images, grayscale_cams, preds, labels)
    ):
        plt.subplot(4, 4, it + 1)
        visualization = show_cam_on_image(image, grayscale_cam)
        plt.imshow(visualization)
        plt.title(f"pred: {pred:.1f} label: {label}")
        plt.axis("off")
    plt.savefig(f"./output/{config.model.name}/class_activation_map_{it}.png")


def visualize_result(config, fold: int):
    path = glob(f"./output/tb_logs/{config.model.name}/*")[-1]
    print(path)
    event_acc = EventAccumulator(path, size_guidance={"scalars": 0})
    event_acc.Reload()

    scalars = {}
    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        scalars[tag] = [event.value for event in events]

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(scalars["lr-AdamW"])), scalars["lr-AdamW"])
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("adamw lr")

    plt.subplot(1, 2, 2)
    plt.plot(
        range(len(scalars["train_loss"])), scalars["train_loss"], label="train_loss"
    )
    plt.plot(range(len(scalars["val_loss"])), scalars["val_loss"], label="val_loss")
    plt.legend()
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.title("train/val rmse")
    plt.savefig(f"./output/{config.model.name}/loss_{fold}.png")
    plt.show()

    # save cv value
    save_path = f"./output/{config.model.name}/{config.expname}"
    if not os.path.exists(save_path):
        os.path.mkdir(save_path)
    save_path = os.path.join(save_path, "cv_results.csv")
    if fold == 0:
        cv_df = pd.DataFrame({"fold_0": min(scalars["val_loss"])})
    else:
        cv_df = pd.read_csv(save_path)
        cv_df[f"fold_{fold}"] = min(scalars["val_loss"])
    cv_df.to_csv(save_path, index=False)


def predict(model, test_dataloader, config, transform):
    preds = []
    for batch in tqdm(test_dataloader):
        img = batch.to(config.device)
        img = transform(img)

        with torch.inference_mode():
            logits = model(img)
        preds.append(logits.sigmoid().cpu().detach().numpy() * 100.0)
    preds = np.concatenate(preds)
    return preds.reshape(-1)


def update_config(config):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fold", default=-1, type=int, nargs="*")
    parser.add_argument("--tpu")
    parser.add_argument("--tpu_cores", default=-10, type=int)
    args = parser.parse_args()

    if args.train_fold != -1:
        assert isinstance(args.train_fold, List)
        assert isinstance(args.train_fold[0], int)
        config["train_epoch"] = args.train_fold
        print("train_epoch is specified: ", config["train_epoch"])

    if args.tpu:
        config["trainer"]["tpu_cores"] = args.tpu_cores
        config["trainer"].pop("gpus")
        print("tpu is specified with the number of", config["trainer"]["tpu_cores"])

    return config


def main(config):
    config = update_config(config)
    if config.train:
        train(config)

    if config.inference:
        sub = []
        test_df = pd.read_csv(os.path.join(config.root, "test.csv"))
        test_df["Id"] = test_df["Id"].apply(
            lambda x: os.path.join(config.root, "test", x + ".jpg")
        )
        test_dataset = PetfinderDataset(
            df=test_df, image_size=config.transform.image_size
        )
        for fold in range(config.n_splits):
            print("\n" + "#" * 8 + f"  Fold: {fold}  " + "#" * 8 + "\n")
            model = Model(config).to(config.device).eval()
            model.load_state_dict(
                torch.load(
                    f"./output/{config.model.name}/best_loss_{fold}.ckpt",
                    map_location=config.device,
                )["state_dict"]
            )

            test_dataloader = DataLoader(test_dataset, **config.test_loader)
            predicts = predict(
                model,
                test_dataloader,
                config,
                transform=get_default_transforms()["test"],
            )
            sub.append(predicts)

        print(np.asarray(sub).shape)
        test_df["Pawpularity"] = np.mean(sub, axis=0)
        print(test_df.head())
        test_df["Id"] = test_df["Id"].apply(lambda x: os.path.basename(x).split(".")[0])
        print(test_df.head())
        submission = test_df[["Id", "Pawpularity"]]
        submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main(config)
