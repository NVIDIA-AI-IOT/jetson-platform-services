# Jetson Platform Services

Jetson Platform Services has several reference AI Services and workflows to build powerful AI applications at the edge such as a VLM alert system and Zero Shot Detection. The AI services and workflows are provided as prebuilt containers with docker compose files hosted on NGC. For comprehensive documentation about Jetson Platform Services, the AI services and workflows, view [the documentation here](https://docs.nvidia.com/jetson/jps) to get started. The linked documentation has all the instructions to setup your Jetson, install Jetson Platform Services and run the AI Services and workflows. 

This repository contains the source code for the reference AI services, workflows and notebooks associated with Jetson Platform Services. **This repository is only needed if you want to customize and rebuild the AI Service containers or run the Jupyter notebook examples.** If you are only looking to run the prebuilt services and workflows, then follow [the documentation here](https://docs.nvidia.com/jetson/jps).

## Setup

To customize and build the AI Service containers locally on your Jetson, you must clone this repository recursively, install Jetson Platform Services and then configure the default docker runtime. 

### Clone Repository (Recursively)
```
git clone --recurse-submodules https://github.com/NVIDIA-AI-IOT/jetson-platform-services
```

### Setup Jetson Services 
Before running any services, workflows or notebooks from this repository, follow the Jetson Platform Services quick-start guide [here](https://docs.nvidia.com/jetson/jps/setup/quick-start.html). The linked quick-start guide will walk you through the steps to setup your Jetson with Jetson Platform Services.

### Docker Configuration

#### Build Runtime

If you are going to build the container locally (instead of using the prebuilt ones from NGC) then the default docker runtime on your Jetson must be set to "nvidia".

On your Jetson, edit the file ```/etc/docker/daemon.json``` to include ```"default-runtime": "nvidia"```
```
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "default-runtime": "nvidia"
}
```
#### Docker Group
If you want to run docker commands without sudo, then add your user to the docker group. 

```
sudo groupadd docker
sudo usermod -aG docker $USER
```

> **_NOTE:_**  After these two steps, reboot your Jetson for it to take effect.

## Repository Contents

With the setup complete, you can now explore the links below to see the source code for the AI services and instructions to rebuild the containers after modification. These containers can then be combined with other parts of Jetson Platform Services to build workflows such as a VLM Alert System. We also provide a sample Jupyter notebook to show how Jetson Platform Services can be used for Traffic Analytics. 

### AI Services

AI services are individual containers that implement a specific AI related function such as Zero Shot Detection or Visual Language Model inference. These AI services can be run on their own and interacted with directly through their respective APIs or combined with other parts of Jetson Platform Services.

1) [Zero Shot Detection (NanoOWL)](inference/zero_shot_detection/README.md)
2) [Visual Language Model (VLM)](inference/vlm/README.md)
3) [DeepStream Perception](inference/perception/README.md)

### Workflows 

Workflows combine several parts of Jetson Platform Services to create full systems such as Zero Shot Detection with dynamic camera routing and a VLM based alert system with mobile app integration.

1) [Zero Shot Detection with SDR](https://docs.nvidia.com/jetson/jps/workflows/zero_shot_detection_workflow.html)
2) [VLM Alert System](https://docs.nvidia.com/jetson/jps/workflows/vlm_workflow.html)

### Notebooks

A sample Jupyter Notebook is provided to show how Jetson Platform Services can be used to generate traffic analytics. 

1) [Traffic Analytics](notebooks/traffic-analytics/README.md)