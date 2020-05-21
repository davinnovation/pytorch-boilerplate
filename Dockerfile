FROM floydhub/pytorch:1.5.0-gpu.cuda10cudnn7-py3.55

RUN mkdir app
WORKDIR app
RUN git clone https://github.com/davinnovation/pytorch-boilerplate
WORKDIR pytorch-boilerplate

RUN pip install -r requirements.txt