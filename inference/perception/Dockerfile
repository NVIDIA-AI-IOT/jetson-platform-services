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

FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch as ds-triton

# Copy model files over
COPY ./dependencies/labels.txt /pn26-files/labels.txt
COPY ./dependencies/resnet34_peoplenet.onnx /pn26-files/resnet34_peoplenet_int8.onnx
COPY ./dependencies/PeopleNet_DEPLOYABLE_v2.6.2_TensorRT_10.3.0_NVIDIA_Jetson_AGX_Orin_Developer_Kit_int8_tao_deploy_utcDateTime_20240816_021742.cal /pn26-files//PeopleNet_calib.cal
COPY ./dependencies/deepstream_tools/yolo_deepstream /yolov8s-files

# Compile necessary engine files for PN2.6
WORKDIR /pn26-files
# For AGX
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":8x3x544x960 --optShapes="input_1:0":8x3x544x960 --maxShapes="input_1:0":8x3x544x960 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs8.engine
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":8x3x544x960 --optShapes="input_1:0":8x3x544x960 --maxShapes="input_1:0":8x3x544x960 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs8.engine
# Additonal for testing
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":9x3x544x960 --optShapes="input_1:0":9x3x544x960 --maxShapes="input_1:0":9x3x544x960 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs9.engine
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":9x3x544x960 --optShapes="input_1:0":9x3x544x960 --maxShapes="input_1:0":9x3x544x960 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs9.engine
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":10x3x544x960 --optShapes="input_1:0":10x3x544x960 --maxShapes="input_1:0":10x3x544x960 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs10.engine
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":10x3x544x960 --optShapes="input_1:0":10x3x544x960 --maxShapes="input_1:0":10x3x544x960 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs10.engine
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":6x3x544x960 --optShapes="input_1:0":6x3x544x960 --maxShapes="input_1:0":6x3x544x960 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs6.engine
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":6x3x544x960 --optShapes="input_1:0":6x3x544x960 --maxShapes="input_1:0":6x3x544x960 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs6.engine

# For NX
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":3x3x544x960 --optShapes="input_1:0":3x3x544x960 --maxShapes="input_1:0":3x3x544x960 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs3.engine
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":3x3x544x960 --optShapes="input_1:0":3x3x544x960 --maxShapes="input_1:0":3x3x544x960 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs3.engine
# Additonal for testing
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":4x3x544x960 --optShapes="input_1:0":4x3x544x960 --maxShapes="input_1:0":4x3x544x960 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs4.engine
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":4x3x544x960 --optShapes="input_1:0":4x3x544x960 --maxShapes="input_1:0":4x3x544x960 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs4.engine
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":2x3x544x960 --optShapes="input_1:0":2x3x544x960 --maxShapes="input_1:0":2x3x544x960 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs2.engine
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":2x3x544x960 --optShapes="input_1:0":2x3x544x960 --maxShapes="input_1:0":2x3x544x960 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs2.engine

# For Nano
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":4x3x544x960 --optShapes="input_1:0":4x3x544x960 --maxShapes="input_1:0":4x3x544x960 --duration=100 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./pn26_jp6_halfmem_bs4_gpu.engine
# Additonal for testing
RUN /usr/src/tensorrt/bin/trtexec --onnx=./resnet34_peoplenet_int8.onnx --calib=./PeopleNet_calib.cal --int8 --minShapes="input_1:0":3x3x544x960 --optShapes="input_1:0":3x3x544x960 --maxShapes="input_1:0":3x3x544x960 --duration=100 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./pn26_jp6_halfmem_bs3_gpu.engine

# Build YOLOv8s DS plugin
RUN apt-get update && apt install -y cmake
WORKDIR /yolov8s-files/deepstream_yolo/nvdsinfer_custom_impl_Yolo/
RUN make && chmod 777 ../build_DLA_engine.sh && ../build_DLA_engine.sh

FROM nvcr.io/nvidia/deepstream:7.1-samples-multiarch as ds-samples
COPY --from=ds-triton /pn26-files/ /pn26-files/
COPY --from=ds-triton /yolov8s-files/deepstream_yolo/nvdsinfer_custom_impl_Yolo/*.so /yolov8s-files/
COPY --from=ds-triton /yolov8s-files/deepstream_yolo/nvdsinfer_custom_impl_Yolo/yolov8s_DAT_640_noqdq.onnx /yolov8s-files/yolov8s_DAT_noqdq.onnx
COPY --from=ds-triton /yolov8s-files/deepstream_yolo/nvdsinfer_custom_impl_Yolo/yolov8s_DAT_640_noqdq_DLA.engine /yolov8s-files/yolov8s_DAT_noqdq_DLA.engine
COPY --from=ds-triton /yolov8s-files/deepstream_yolo/nvdsinfer_custom_impl_Yolo/yolov8s_DAT_precision_config_calib.cache /yolov8s-files/yolov8s_DAT_precision_config_calib.cache

# Compile Service Maker test5
RUN apt-get update && apt install -y cmake
RUN mkdir /opt/nvidia/deepstream/deepstream/service-maker/sources/apps/cpp/deepstream_test5_app/build
WORKDIR /opt/nvidia/deepstream/deepstream/service-maker/sources/apps/cpp/deepstream_test5_app/build
RUN cmake .. && make

COPY ./dependencies/LICENSE_thirdparty.txt /opt/nvidia/deepstream/deepstream/LICENSE_thirdparty.txt
WORKDIR /opt/nvidia/deepstream/deepstream/
