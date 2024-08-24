import numpy as np
import argparse

import torch

from torch.utils.data import DataLoader

import config
from dataset.charts_image_dataset import ChartImageTypeDataset
from models.SimpleCNNModel import SimpleCNNModel
from training import ProgressCounter
from training.CustomEvaluatorTrainer import CustomTrainer, CustomEvaluator
from training.Manager import JsonWandBManager, JsonManager
from training.train import train_model


def run(args):

    # load data
    train_dataset = ChartImageTypeDataset(dataset_part="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = ChartImageTypeDataset(dataset_part="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    val_dataset = test_dataset
    val_dataloader = test_dataloader

    # defined_models = [SimpleCNNModel, SimpleCNNModelBig, SimpleCNNModelSecondDimension, SimpleCNNModelSecondDimensionBig]
    name = f"{args.name}_model_SimpleCNNModel_lr{args.lr}_epochs{args.epochs}"
    number_of_classes = len(config.dataset_info.chart_type_labels)

    model = SimpleCNNModel(
        device=None, # torch.device("cpu"),
        out_dim=number_of_classes,
        dropout=args.dropout
    )

    # if one class should have different weight for loss function
    # example class B will have different weight contrary to the other classes
    # default = all classes same weight
    classw_tensor = torch.tensor([1.] * len(config.dataset_info.chart_type_labels))


    loss_function = torch.nn.CrossEntropyLoss(weight=classw_tensor.to(model.device))
    optimizer = torch.optim.AdamW(model.parameters())
    run_config = {
        "dataset": train_dataset.__class__.__name__,
        "model": model.__class__.__name__,
        "optimizer": str(optimizer),
        "args": args.__dict__
    }

    if args.wandb:
        mngr = JsonWandBManager(name, run_config)
    else:
        mngr = JsonManager(name, run_config)

    # create a trainer object
    trainer = CustomTrainer(model, train_dataloader, optimizer, loss_function=loss_function, log_manager=mngr,
                            log_each_step=10, gradient_accumulation_steps=args.grad_accum_steps)
    val_evaluator = CustomEvaluator(model, val_dataloader, loss_function=loss_function, log_manager=mngr)
    test_evaluator = CustomEvaluator(model, test_dataloader, loss_function=loss_function, log_manager=mngr)

    train_model(trainer, val_evaluator, test_evaluator, epochs=args.epochs, save_metric_max_keys=["macro_f1", "acc"])

    mngr.close()


if __name__ == '__main__':

    # some info:
    #     please add some hint in the name of run (e.g. model used, difference vs default):
    #     python main.py --wandb --name someHint    #
    #     to try memory fit, set "--warm_epochs 0" so the model is not frozen
    #     If you change the code: do also commit, push into new branch (so it can be found in wandb) and mention that in --name arg

    parser = argparse.ArgumentParser(description='Chart type Image experiments -- dataset icpr2022_chart_image_dataset')
    parser.add_argument('--name', default='chart_type_image_experiment', type=str, help='Identifier of the run.')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--grad_accum_steps', default=0, type=int, help='gradient accumulation steps <= 1 means no grad. accum.')

    parser.add_argument("--dropout", default=0.1, type=float)

    parser.add_argument('--wandb', action='store_true', help='enable wandb logging')

    args = parser.parse_args()
    run(args)
