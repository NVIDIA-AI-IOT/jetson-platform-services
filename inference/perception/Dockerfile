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

FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch as ds-triton

# Copy model files over
COPY ./dependencies/Tracker/ /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
COPY ./dependencies/peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache /pn26-files/peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache
COPY ./dependencies/peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff /pn26-files/peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff
COPY ./dependencies/labels.txt /pn26-files/labels.txt

# Compile necessary engine files for PN2.6
WORKDIR /pn26-files
# For AGX
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=8 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs8-8623.engine
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=8 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs8-8623.engine
# Additonal for testing
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=9 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs9.engine
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=9 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs9.engine
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=10 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs10.engine
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=10 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs10.engine
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=6 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs6.engine
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=6 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs6.engine

# For NX16
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=3 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs3.engine
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=3 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs3.engine
# Additonal for testing
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=4 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs4.engine
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=4 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs4.engine
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=2 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs2.engine
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=2 --duration=100 --useDLACore=1 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla1_pn26_jp6_halfmem_bs2.engine

# For NX8
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=4 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs4.engine
# Additonal for testing
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=8 --duration=100 --useDLACore=0 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./dla0_pn26_jp6_halfmem_bs8.engine

# For Nano
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=4 --duration=100 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./pn26_jp6_halfmem_bs4_gpu.engine
# Additonal for testing
RUN /usr/src/tensorrt/bin/trtexec --uff=./peoplenet_V2.6_resnet34_hr_noreg_ph2_dla.uff --int8 --uffInput=input_1,3,544,960 --output=output_bbox/BiasAdd,output_cov/Sigmoid --calib=./peoplenet_V2.6_resnet34_hr_noReg_ph2_dla.cache --batch=3 --duration=100 --allowGPUFallback --memPoolSize=dlaSRAM:0.5 --verbose --saveEngine=./pn26_jp6_halfmem_bs3_gpu.engine


RUN apt update && apt install -y cmake
RUN mkdir /opencv-temp && cd /opencv-temp
WORKDIR /opencv-temp
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
RUN unzip opencv.zip && unzip opencv_contrib.zip
RUN mkdir -p build
WORKDIR /opencv-temp/build
RUN cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules -D OPENCV_GENERATE_PKGCONFIG=YES ../opencv-4.x
RUN cmake --build . -j 8
RUN make install
RUN rm -rf /opencv-temp/build

# Build YOLOv8s DS plugin
RUN mkdir /yolov8s-files
WORKDIR /yolov8s-files
RUN git clone https://github.com/marcoslucianops/DeepStream-Yolo
RUN cd DeepStream-Yolo && CUDA_VER=12.2 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
RUN mv DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so . && rm -rf DeepStream-Yolo

FROM nvcr.io/nvidia/deepstream:7.0-samples-multiarch as ds-samples
COPY --from=ds-triton /opt/nvidia/deepstream/deepstream/samples/models/Tracker/ /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
COPY --from=ds-triton /pn26-files/ /pn26-files/
COPY --from=ds-triton /yolov8s-files/ /yolov8s-files/

# Compile Service Maker test5
RUN apt update && apt install -y cmake
RUN mkdir /opt/nvidia/deepstream/deepstream-7.0/service-maker/sources/apps/deepstream_test5_app/build
WORKDIR /opt/nvidia/deepstream/deepstream-7.0/service-maker/sources/apps/deepstream_test5_app/build
RUN cmake .. && make

# Install OpenCV dependency for YOLOv8s
COPY --from=ds-triton /opencv-temp/ /opencv-temp/
RUN mkdir -p /opencv-temp/build
WORKDIR /opencv-temp/build
RUN cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules -D OPENCV_GENERATE_PKGCONFIG=YES ../opencv-4.x
RUN cmake --build . -j 8
RUN make install
RUN ldconfig
WORKDIR /opt/nvidia/deepstream/deepstream-7.0
RUN rm -rf /opencv-temp

COPY ./dependencies/yolov8s/yolov8s_setup.sh /yolov8s-files/yolov8s_setup.sh
COPY ./dependencies/LICENSE_thirdparty.txt /opt/nvidia/deepstream/deepstream-7.0/LICENSE_thirdparty.txt
