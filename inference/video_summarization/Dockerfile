# SPDX-License-Identifier: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/e7ep4mig3lne/nvidia-dev/vlm_infer:2.0.0

USER root
WORKDIR /jetson-inference

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANGUAGE=en_US:en \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get install -y --no-install-recommends python3-pip git-all cudnn9-cuda-12

# Install Python packages
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-dependencies --no-cache-dir -r requirements-jetson.txt

RUN mkdir -p /data/models/mlc/dist/models

WORKDIR /
RUN wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/7.1/files?redirect=true&path=deepstream_sdk_v7.1.0_jetson.tbz2' -O deepstream_sdk_v7.1.0_jetson.tbz2
RUN tar -xvf deepstream_sdk_v7.1.0_jetson.tbz2 -C / && rm deepstream_sdk_v7.1.0_jetson.tbz2
RUN cd /opt/nvidia/deepstream/deepstream-7.1 && ./install.sh && ldconfig

RUN apt-get -y purge python3-pip git-all
COPY . /video_summarization/
WORKDIR /video_summarization

CMD ["python3", "/video_summarization/src/api_server.py"]
