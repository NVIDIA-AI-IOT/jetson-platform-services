# Visual Language Model (VLM)

The primary documentation for the VLM service can be found [here](https://docs.nvidia.com/jetson/jps/inference-services/vlm.html)

It is recommend to first go through the primary documentation to learn how to run the VLM service. This README will cover how to customize and rebuild the container.

## Customize

The VLM Service has the following python files:

- main.py - Main pipeline that pulls together streaming, model inference and API server interaction 

- chat_server.py - Internal service that runs and exposes an OpenAI like chat server for the VLM. The main.py script will use the chat server for VLM inference on frames from the added live stream. 

- ws_server.py - Web Socket Server that will output alert states. Alert states are pushed from main.py and then output to all clients connected to the web socket such as a mobile app for notifications. 

- utils.py - Miscellaneous utility functions called by main.py

- config.py - Defines configuration options for this microservice. Modify his file to add new configuration options. Options here corresponds to the config/main_config.json file 


Many of the common components that can be used for building new microservices have been broken out into modular components contained within the [mmj-utils library](https://github.com/NVIDIA-AI-IOT/mmj_utils) such as RTSP streaming, overlay generation, VLM interaction and the base API Server.

After modifying the source code, you can test the changes by launching the container with the compose-dev.yaml file 

```
cd ~/jetson-platform-services/inference/vlm
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

Navigate to the ```vlm``` directory 

```
cd ~/jetson-platform-services/inference/vlm
sudo bash build_container.sh 
```

The build container script will rebuild the container with the modified source code.

You can then launch the container with the compose.yaml file 

```
cd ~/jetson-platform-services/inference/vlm
sudo docker compose up
```

It can also be launched from the workflow examples 

```
cd ~/jetson-platform-services/ai_service_workflow/vlm/example_1/
sudo docker compose up 
```