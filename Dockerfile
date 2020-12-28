FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN mkdir app
WORKDIR app
RUN git clone https://github.com/davinnovation/pytorch-boilerplate
WORKDIR pytorch-boilerplate

RUN pip install -r requirements.txt