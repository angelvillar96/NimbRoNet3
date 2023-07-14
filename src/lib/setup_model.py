"""
Setting up the model, encoder backbones, optimizers, loss functions,
loading/saving parameters, ...
"""

import os
import traceback
import torch

from lib.logger import log_function, print_
from lib.schedulers import LRWarmUp, ExponentialLRSchedule
from lib.utils import create_directory
import models
from models.model_utils import split_model_parameters
from models.Encoders import ResNet18, ResNet34, ResNet50, EfficientnetB0, MobileNetV3
from CONFIG import MODELS, ENCODERS


@log_function
def setup_model(model_params):
    """
    Loading the model given the model parameters stated in the exp_params file

    Args:
    -----
    model_params: dictionary
        model parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    model: torch.nn.Module
        instanciated model given the parameters
    """
    model_name = model_params["model_name"]
    if model_name not in MODELS:
        raise NotImplementedError(f"Model '{model_name}' not in recognized models: {MODELS}")
    cur_model_params = model_params.get(model_name, {})

    # Detection and Segmentation
    if(model_name == "NimbroNetV2"):
        model = models.NimbroNetV2(**cur_model_params)
    elif (model_name == "NimbroNetV2plus"):
        encoder, _ = setup_encoder(model_params)
        model = models.NimbroNetV2plus(encoder=encoder, **cur_model_params)
    elif (model_name == "NimbroNetV3"):
        encoder, _ = setup_encoder(model_params)
        model = models.NimbroNetV3(encoder=encoder, **cur_model_params)        
    else:
        raise NotImplementedError(f"Model '{model_name}' not in recognized models: {MODELS}")

    return model


@log_function
def setup_encoder(model_params):
    """ Loading the backbone encoder given encoder name """
    model_name = model_params.get("backbone", "ResNet18")
    pretrained = model_params.get("pretrained", True)
    if model_name not in ENCODERS:
        raise NotImplementedError(f"Encoder '{model_name}' not in recognized encoders: {ENCODERS}")

    if model_name == "ResNet18":
        encoder = ResNet18(pretrained=pretrained)
    elif model_name == "ResNet34":
        encoder = ResNet34(pretrained=pretrained)
    elif model_name == "ResNet50":
        encoder = ResNet50(pretrained=pretrained)
    elif model_name == "EfficientNetB0":
        encoder = EfficientnetB0(pretrained=pretrained)
    elif model_name == "MobileNetv3":
        encoder = MobileNetV3(pretrained=pretrained)
    else:
        raise NotImplementedError(f"Encoder '{model_name}' not in recognized encoders: {ENCODERS}")

    return encoder, pretrained


def emergency_save(f):
    """
    Decorator for saving a model in case of exception, either from code or triggered.
    Use for decorating the training loop:
        @setup_model.emergency_save
        def train_loop(self):
    """

    def try_call_except(*args, **kwargs):
        """ Wrapping function and saving checkpoint in case of exception """
        try:
            return f(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            print_("There has been an exception. Saving emergency checkpoint...")
            self_ = args[0]
            if hasattr(self_, "model") and hasattr(self_, "optimizer"):
                fname = f"emergency_checkpoint_epoch_{self_.epoch}.pth"
                save_checkpoint(
                    model=self_.model,
                    optimizer=self_.optimizer,
                    scheduler=self_.scheduler,
                    epoch=self_.epoch,
                    exp_path=self_.exp_path,
                    savedir="models",
                    savename=fname
                )
                print_(f"  --> Saved emergency checkpoint {fname}")
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()

    return try_call_except


@log_function
def save_checkpoint(trainer, finished=False, savedir="models", savename=None):
    """
    Saving a checkpoint in the models directory of the experiment. This checkpoint
    contains state_dicts for the mode, optimizer and lr_scheduler
    """

    if(savename is not None):
        checkpoint_name = savename
    elif(savename is None and finished is True):
        checkpoint_name = "checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{trainer.epoch}.pth"

    create_directory(trainer.exp_path, savedir)
    savepath = os.path.join(trainer.exp_path, savedir, checkpoint_name)

    scheduler_data = "" if trainer.scheduler is None else trainer.scheduler.state_dict()
    torch.save({
            'epoch': trainer.epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            "scheduler_state_dict": scheduler_data
            }, savepath)

    return


@log_function
def load_checkpoint(checkpoint_path, model, only_model=False, map_cpu=False, **kwargs):
    """
    Loading a precomputed checkpoint: state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    checkpoint_path: string
        path to the .pth file containing the state dicts
    model: torch Module
        model for which the parameters are loaded
    only_model: boolean
        if True, only model state dictionary is loaded
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist ...")
    if(checkpoint_path is None):
        return model

    # loading model to either cpu or cpu
    if(map_cpu):
        checkpoint = torch.load(checkpoint_path,  map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path)
    # loading model parameters. Try catch is used to allow different dicts
    if "model_state_dict" in checkpoint.keys():
        model_dict = checkpoint['model_state_dict']
       # if "segmentation_head.weight" not in model_dict:  # denoised pretrained model
          #  model_dict = add_dummy_head_params(model_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint)

    # returning only model for transfer learning or returning also optimizer for resuming training
    if(only_model):
        return model

    optimizer, scheduler = kwargs["optimizer"], kwargs["scheduler"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint["epoch"] + 1

    return model, optimizer, scheduler, epoch


@log_function
def setup_optimization(exp_params, model):
    """
    Initializing the optimizer object used to update the model parameters

    Args:
    -----
    exp_params: dictionary
        parameters corresponding to the different experiment
    model: nn.Module
        instanciated neural network model
    Returns:
    --------
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    """
    lr = exp_params.get("lr", 1e-3)
    lr_encoder = exp_params.get("lr_encoder", 1e-4)

    # filtering parameters to assign different learning rates
    model_params, encoder_params = split_model_parameters(model)
    parameters = [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": model_params, "lr": lr}
        ]
    if lr_encoder != lr:
        print_(f"  --> Model learning rate {lr}")
        print_(f"  --> Encoder learning rate {lr_encoder}")

    # setting up optimizer and LR-scheduler
    optimizer = setup_optimizer(parameters, exp_params)
    scheduler = setup_scheduler(exp_params, optimizer)

    return optimizer, scheduler


def setup_optimizer(parameters, exp_params):
    """ Instanciating a new optimizer """
    lr = exp_params.get("lr", 1e-3)
    momentum = exp_params.get("momentum", False)
    optimizer = exp_params.get("optimizer", "adam")
    nesterov = exp_params.get("nesterov", False)
    weight_decay = exp_params.get("weight_decay", 0.0001)

    # SGD-based optimizer
    if(optimizer == "adam"):
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif(optimizer == "adamw"):
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                                    nesterov=nesterov, weight_decay=weight_decay)

    return optimizer


def setup_scheduler(exp_params, optimizer):
    """ Instanciating a new scheduler """
    lr = exp_params.get("lr", 1e-3)
    lr_factor = exp_params.get("lr_factor", 1.)
    patience = exp_params.get("patience", 10)
    scheduler = exp_params.get("scheduler", "")

    if(scheduler == "plateau"):
        print_("Setting up Plateau LR-Scheduler:")
        print_(f"  --> Patience: {patience}")
        print_(f"  --> Factor:   {lr_factor}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=patience,
                factor=lr_factor,
                min_lr=1e-8,
                mode="min",
                verbose=True
            )
    elif(scheduler == "step"):
        print_("Setting up Step LR-Scheduler")
        print_(f"  --> Step Size: {patience}")
        print_(f"  --> Factor:    {lr_factor}")
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                gamma=lr_factor,
                step_size=patience
            )
    elif(scheduler == "exponential"):
        print_("Setting up Exponential LR-Scheduler")
        print_(f"  --> Init LR: {lr}")
        print_(f"  --> Factor:  {lr_factor}")
        scheduler = ExponentialLRSchedule(
                optimizer=optimizer,
                init_lr=lr,
                gamma=lr_factor
            )
    else:
        print_("Not using any LR-Scheduler")
        scheduler = None

    return scheduler


def update_scheduler(scheduler, exp_params, control_metric=None, iter=-1, end_epoch=False):
    """
    Updating the learning rate scheduler

    Args:
    -----
    scheduler: torch.optim
        scheduler to evaluate
    exp_params: dictionary
        dictionary containing the experiment parameters
    control_metric: float/torch Tensor
        Last computed validation metric.
        Needed for plateau scheduler
    iter: float
        number of optimization step.
        Needed for cyclic, cosine and exponential schedulers
    end_epoch: boolean
        True after finishing a validation epoch or certain number of iterations.
        Triggers schedulers such as plateau or fixed-step
    """
    if scheduler is None:
        return

    scheduler_type = exp_params["training"].get("scheduler", "")
    if(scheduler_type == "plateau" and end_epoch):
        scheduler.step(control_metric)
    elif(scheduler_type == "step" and end_epoch):
        scheduler.step()
    elif(scheduler_type == "exponential"):
        scheduler.step(iter)
    return


@log_function
def setup_lr_warmup(params):
    """
    Seting up the learning rate warmup handler given experiment params

    Args:
    -----
    params: dictionary
        training parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    lr_warmup: Object
        object that steadily increases the learning rate during the first iterations.

    Example:
    -------
        #  Learning rate is initialized with 3e-4 * (1/1000). For the first 1000 iterations
        #  or first epoch, the learning rate is updated to 3e-4 * (iter/1000).
        # after the warmup period, learning rate is fixed at 3e-4
        optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
        lr_warmup = LRWarmUp(init_lr=3e-4, warmup_steps=1000, max_epochs=1)
        ...
        lr_warmup(iter=cur_iter, epoch=cur_epoch, optimizer=optimizer)  # updating lr
    """
    use_warmup = params["lr_warmup"]
    lr = params["lr"]
    if(use_warmup):
        warmup_steps = params["warmup_steps"]
        warmup_epochs = params["warmup_epochs"]
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=warmup_steps, max_epochs=warmup_epochs)
        print_("Setting up learning rate warmup:")
        print_(f"  --> Target LR:     {lr}")
        print_(f"  --> Warmup Steps:  {warmup_steps}")
        print_(f"  --> Warmup Epochs: {warmup_epochs}")
    else:
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=-1, max_epochs=-1)
        print_("Not using learning rate warmup...")
    return lr_warmup


#
