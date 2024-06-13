# DeepStream (DS) Perception

The Docker container built using the Dockerfile in this folder can be found on NGC: nvcr.io/nvidia/jps/deepstream:7.0-jps-v1.1.1

If instead, you would like to build it yourself, follow the steps below:
## Building the container

Ensure you have followed the [setup steps](./../../README.md#setup)

To build, run the Docker build command from this folder, specifying the tag as desired:

```
sudo docker build -t deepstream-jps:latest .
```

Note that this needs to be done on a Jetson due to the fact that the Dockerfile is building engine files. This process will take about 45 minutes to an hour to complete on a Jetson Orin AGX. This Dockerfile first builds engine files for various batch sizes for different devices, then installs dependencies needed for YOLOv8 such as OpenCV. It also compiles a third party OSS DeepStream library for use with YOLOv8 and compiles the DeepStream Service Maker test5 application for use with the AI-NVR JPS sample application.

## Using the container
Sample configs for both DeepStream and Docker Compose are provided with the sample applications distributed with Jetson Platform Services (JPS). Links for these can be found on the JPS docs: https://docs.nvidia.com/jetson/jps/setup/quick-start.html

These docs also provide info on deploying with PeopleNet2.6 and YOLOv8s. PeopleNet2.6 deployment is part of the quick start guide linked above while YOLOv8s deployment steps can be found under the DeepStream section of the docs: https://docs.nvidia.com/jetson/jps/deepstream/deepstream.html#yolov8s-deployment
