# Training

We provide our entire pipeline for training our NimbRoNet models for object detection, semantic segmentation, and robot
pose estimation on the soccer field.


## Train NimbRoNet model

**1.** Create a new experiment using the `src/01_create_experiment.py` script. This will create a new experiments folder in the `/experiments` directory.

```
usage 01_create_experiment.py [-h] -d EXP_DIRECTORY [--name NAME] [--config CONFIG]

optional arguments:
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY Directory where the experiment folder will be created
  --name NAME           Name to give to the experiment
```



**2.** Modify the experiment parameters located in `experiments/YOUR_EXP_DIR/YOUR_EXP_NAME/experiment_params.json` to adapt to your dataset and training needs.
We provide one [example](https://github.com/angelvillar96/NimbRoNet3/blob/master/src/configs/nimbronet3.json) for training our NimbRoNet3.

For instance, you can change the `/model/model_name` parameter to choose which model type you want to train, e.g., NimbRoNet2, NimbRoNet2+, or NimbRoNet3.



**3.** Train the specified NimbRoNet model given the specified experiment parameters using the `src/02_train_nimbronetv2.py` or `src/02_train_nimbronetv3.py`
scripts.

```
usage: 02_train_nimbronetV2.py [-h] -d EXP_DIRECTORY [--checkpoint CHECKPOINT] [--resume_training]

optional arguments:
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Path to the experiment directory
  --checkpoint CHECKPOINT
                        Checkpoint with pretrained parameters to load
  --resume_training     For resuming training
```


The training can be monitored using Tensorboard.
To launch tensorboard:

```
tensorboard --logdir experiments/EXP_DIR/EXP_NAME --port 8888
```


**4.** Evaluate a trained model using the `src/03_evaluate_nimbronetv2.py` or `src/03_evaluate_nimbronetv3.py`
scripts.

usage: 03_evaluate_nimbronetv3.py [-h] -d EXP_DIRECTORY [--checkpoint CHECKPOINT]

```
optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY
                        Path to the experiment directory
  --checkpoint CHECKPOINT
                        Checkpoint with pretrained parameters to load
```



### Example: NimbRoNet3 Training

Below we provide an example of how to train a new NimbRoNet3 model:

```
python src/01_create_experiment.py -d new_exps --name my_exp
python src/02_train_nimbronetv3.py -d experiments/new_exps/my_exp
python src/03_evaluate_nimbronetv3.py -d experiments/new_exps/my_exp --checkpoint checkpoint_epoch_30.pth
```
