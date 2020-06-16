# pytorch-boilerplate : flashlight

The OTHER pytorch boilerplate.

![Untitled](https://user-images.githubusercontent.com/3917185/84566705-4a425f80-adae-11ea-92f5-6290aff0478a.png)

## Pre-requirements for local [*PRTM!*]

- python 3.6
- pytorch 1.5.0, torchvision 0.6.0 for your OS/CUDA match version
- ... and install requirements.txt packages `pip install -r requirements.txt`

## Getting Started

`master` branch runs MNIST classification (torchvision dataset) with squeezenet (torchvision model)
for detail, check `config/config.py`

### Run Single Experiment without NNI

0. Prepare enviroment : gpu docker, local python env... whatever

- if docker : `docker pull davinnovation/pytorch-boilerplate:alpha`

1. `python run.py`

![image](https://user-images.githubusercontent.com/3917185/84721592-fc9b4200-afbb-11ea-9602-c41dc58f8b8a.png)

2. after experiment... `tensorboard --logdir Logs`

![image](https://user-images.githubusercontent.com/3917185/84721667-26ecff80-afbc-11ea-8152-4025cbaeda90.png)

### Run Experiments with NNI

0. Prepare environment

1. `nnictl create --config nni_config.yml`

2. localhost:8080

![image](https://user-images.githubusercontent.com/3917185/84721734-484deb80-afbc-11ea-8585-60f1752dd1d8.png)

## Diving into Code

![image](https://user-images.githubusercontent.com/3917185/84723043-ac25e380-afbf-11ea-9116-fbabd47b5cc0.png)

- Adding Network

`flashlight.network.__init__.py`

![image](https://user-images.githubusercontent.com/3917185/84722900-61a46700-afbf-11ea-9fda-9a5801eb2c1e.png)

- Adding Dataset

`flashlight.dataloader.__init__.py`

![image](https://user-images.githubusercontent.com/3917185/84722792-2f930500-afbf-11ea-9ea1-3bbe25905d96.png)

- Change Loss, forward/backward... [Research Code]

`flashlight.runner.pl.py`

- Change Logger, hw options... [Engineering Code]

`flashlight.runner.main_pl.py`