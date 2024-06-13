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

#! /bin/bash -e

echo "NOTE: For each dataset an user elects to use, the user is responsible for checking if the dataset license is fit for the intended purpose."

cd /yolov8s

INT8_CALIB_IMG_PATH="/yolov8s/yolov8s-dependencies/calibration.txt"
INT8_CALIB_BATCH_SIZE=4
action="$1"

case "$action" in
   --agx)
       echo "Using AGX so batch size set to 8..."
       INT8_CALIB_BATCH_SIZE=8
       ;;
   --nx16)
       echo "Using NX16 so batch size set to 4..."
       INT8_CALIB_BATCH_SIZE=4
       ;;
   *)
       echo "Unknown option '$action'. Exiting..."
       exit 1
       ;;
esac

export INT8_CALIB_IMG_PATH
export INT8_CALIB_BATCH_SIZE

if [ ! -d "./yolov8s-dependencies" ]; then
    mkdir yolov8s-dependencies
    cd yolov8s-dependencies
    mkdir calibration

    # Download images used for calibration file creation
    wget http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
    unzip val2017.zip
    for jpg in $(ls -1 val2017/*.jpg | sort -R | head -1000); do \
        cp ${jpg} calibration/; \
    done

    find calibration -type f -printf '/yolov8s/yolov8s-dependencies/%p\n' > calibration.txt

    # Install dependencies for script to convert .pt to .onnx
    apt update
    apt install python3-dev -y
    pip3 install onnx onnxsim onnxruntime ultralytics torch

    # Download YOLOv8s model and conversion script
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
    wget https://raw.githubusercontent.com/marcoslucianops/DeepStream-Yolo/a9d2f135443c16fbf1a08bc763a17bf3b55f608c/utils/export_yoloV8.py

    # Convert .pt to .onnx to use with DS
    python3 export_yoloV8.py -w yolov8s.pt --dynamic

else
    echo "yolov8s-dependencies directory exists - skipping download/init. If there was a failure last time, please delete this folder and retry."
fi


# Run DS to create engine and calibration file for later use
cd $(dirname "$0")

case "$action" in
   --agx)
       echo "Running AGX setup..."
       deepstream-test5-app -c /ds-config-files/yolov8s/setup_config_agx.txt
       ;;
   --nx16)
       echo "Running NX16 setup..."
       deepstream-test5-app -c /ds-config-files/yolov8s/setup_config_nx16.txt
       ;;
   *)
       echo "Unknown option '$action'. Skipping engine/calib file creation..."
       exit 1
       ;;
esac

cp /yolov8s-files/*.engine /yolov8s/
