# A CPU-friendly real-time face tracker using DeepSORT and OpenVINO

This project serves as a proof of concept of how we can se a strong and well known algorithm for multi-object tracking,
the [DeepSORT](https://arxiv.org/abs/1703.07402) (Deep Simple Online Realtime Tracking) algorithm, which the biggest different from their antecesor is the "Deep" part, which is realted to the used of a pre-trained deep learning model as a image encoder to improve the re-identification capability of the original solution (SORT). Even though the DeepSORT is classified as a Realtime solution for multi-object tracking (MOT), because it needs to run a deep learning encoder to extract those features per object (usually deep CNN pre-trained on Imagenet challenge), the overall solution has a natural bottleneck in that part, and because in the real-life problems, it's always better to achieave real-time (at least 30 fps) using only CPU, to reduce costs and to fit the edge machine limitations.

With all said, this POC aimed to show how to use OpenVINO to reduce the bottleneck on the "deep" part of the DeepSORT, using a pretrained Mobilenetv3-large in the "OpenVINO IR" format (OpenVINO optimized for inference format) and as a object detector we a going to use a really fast pre-trained face detection avaible on the OpenVINO model zoo.

Highlighting a few advantages of this approachs:
* CPU friendly solution, keeping real-time capability;
* Light python dependences, do not need to install pytorch and other heave deeplearning frameworks to make use of the DeepSort lib.
* Flexible solution, it's easy to change the encoder, here we are using the MobilenetV3-large, but we can change to use any other, a lither or even a heaver
one, but can be more precise to re-ID process, such as ConvNexT for instance.

# Creating the python environment

## Conda
```
$ conda create -n face-tracker python=3.9 -y
$ conda activate face-tracker
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
```

## Pipy
```
$ python -m venv venv
$ source venv/bin/activate
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
```