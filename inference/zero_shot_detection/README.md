# Zero Shot Detection

The primary documentation for the Zero Shot detection service can be found [here](https://docs.nvidia.com/jetson/jps/inference-services/zero_shot_detection.html)

It is recommend to first go through the primary documentation to learn how to run the Zero Shot Detection service. This README will cover how to customize and rebuild the container.

## Customize 

The Zero Shot Detection Service is built with the following python files:

- main.py - Main pipeline that pulls to together the streaming, model inference and API Server interaction. 

- api_server.py - Defines the REST API server with FastAPI. Customize this to add new API endpoints. Inherits from the api server defined in [mmj-utils library](https://github.com/NVIDIA-AI-IOT/mmj_utils). 

- config.py - Defines configuration options for this microservice. Modify this file to add new configuration options. Options are set in the config/main_config.json file.

Many of the common components that can be used for building new microservices have been broken out into modular components contained within the [mmj-utils library](https://github.com/NVIDIA-AI-IOT/mmj_utils) such as RTSP streaming, overlay generation, redis output and the base API Server.

After modifying the source code, you can test the changes by launching the container with the compose-dev.yaml file 

```
cd ~/jetson-services/inference/zero_shot_detection
docker compose -f compose-dev.yaml up 
```

This will launch the prebuilt container from NGC, and mount the changes made in the source code into the container. This allows the container to run with the modified source code to rapidly test changes without rebuilding the container. 


After testing your changes, the container can be brought back down. 

```
docker compose -f compose-dev.yaml down 
```

## Build Container

To make the changes persist, the container can be rebuilt to include the modifications. 

First ensure you have followed the [docker setup steps](./../../README.md#setup)

Navigate to the ```zero_shot_detection``` directory and run the build script.

```
cd ~/jetson-services/inference/zero_shot_detection
sudo bash build_container.sh 
```

The build container script will rebuild the container with the modified source code.

You can then launch the container with the compose.yaml file 

```
cd ~/jetson-services/inference/zero_shot_detection
sudo docker compose up
```

It can also be launched from the workflow examples 

```
cd ~/jetson-services/ai_service_workflow/zero_shot_detection/example_1/
sudo docker compose up 
```