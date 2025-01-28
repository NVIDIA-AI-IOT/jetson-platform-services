# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pydantic_settings import BaseSettings
import json 

class AppConfig(BaseSettings):
    api_server_port: int
    stream_output: str
    redis_host: str
    redis_port: int 
    redis_stream: str 
    redis_output_interval: int
    model: str = "google/owlvit-base-patch32"
    image_encoder_engine: str = "/data/owl_image_encoder_patch32.engine"
    log_level: str 

def load_config(config_path: str) -> AppConfig:
    with open(config_path, 'r') as file:
        config_data = json.load(file)
    return AppConfig(**config_data)