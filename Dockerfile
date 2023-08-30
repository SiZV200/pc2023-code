FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN pip3 install -U scikit-learn
RUN pip3 install -U tqdm
RUN pip3 install -U mne

## Include the following line if you have a requirements.txt file.
## RUN pip install -r requirements.txt
