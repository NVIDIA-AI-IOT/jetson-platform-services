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
from queue import Queue
import logging 
import os 
from nanoowl.owl_predictor import OwlPredictor
from config import load_config 
from mmj_utils.schema_gen import SchemaGenerator
from mmj_utils.overlay_gen import DetectionOverlayCUDA
from mmj_utils.streaming import VideoOutput, VideoSource
from jetson_utils import cudaResize, cudaAllocMapped

from api_server import DetectionServer

#Helper function to process prompt inputs
def process_prompt(prompt):
    objects = prompt["objects"]
    objects = [x.strip() for x in objects]
    thresholds = prompt["thresholds"]
    thresholds = [float(x) for x in thresholds]
    return objects, thresholds 

def resize_bboxes(bboxes, original_size, new_size):
    original_width, original_height = original_size
    new_width, new_height = new_size
    
    # Calculate the scaling factors
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    
    # Resize each bounding box
    resized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        new_x1 = int(x1 * scale_x)
        new_y1 = int(y1 * scale_y)
        new_x2 = int(x2 * scale_x)
        new_y2 = int(y2 * scale_y)
        resized_bboxes.append([new_x1, new_y1, new_x2, new_y2])
    
    return resized_bboxes


#Load config
config_path = os.environ["MAIN_CONFIG_PATH"] #TODO connect the configs 
config = load_config(config_path)

logging.basicConfig(level=logging.getLevelName(config.log_level),
                    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

#Communication between flask and main loop 
cmd_q = Queue()
cmd_resp = dict()

#Launch flask server and connect queue to receive prompt updates 
api_server = DetectionServer(cmd_q, cmd_resp, port=config.api_server_port)
api_server.start_server()

#RTSP input and output 
v_input = VideoSource() #start as None will connect upon api request
v_output = VideoOutput(config.stream_output)

#overlay generation and redis connection 
overlay_gen = DetectionOverlayCUDA(max_objects=10) #used to generate bounding box overlays 
schema_gen = SchemaGenerator(sensor_id=1, sensor_type="camera", sensor_loc=[10,20,30]) #used to generate metadata output in metropolis minimal schema format 
schema_gen.connect_redis(config.redis_host, config.redis_port, config.redis_stream) #connect schema output to redis stream

#Load GenAI model
predictor = OwlPredictor(
    config.model,
    image_encoder_engine=config.image_encoder_engine
)

# create video input and output
v_input = VideoSource() #start as None will connect upon api request
v_output = VideoOutput(config.stream_output)

frame_counter = 0
skip_counter = 0
objects = None 
thresholds = None 
while(True):

    #Get commands from flask
    if not cmd_q.empty():
        message = cmd_q.get()

        if message.type == "classes":
            objects, thresholds = process_prompt(message.data)
            if len(objects) > 0 and len(thresholds) > 0:
                objects_encoding = predictor.encode_text(objects)
                cmd_resp[message.id] = "Detection classes set."
            else:
                objects = None 
                thresholds = None 
                cmd_resp[message.id] = "Detection classes cleared."
            
        elif message.type == "stream_add":
                logging.debug("Message is a stream add")

                #cannot add stream 
                if v_input.connected:
                    cmd_resp[message.id] = {"success": False, "message": "Stream Maximum reached. Remove a stream first to add another."}
                
                #adds stream
                else:
                    #add new stream 
                    rtsp_link = message.data["stream_url"]
                    v_input.connect_stream(rtsp_link, camera_id=message.data["stream_id"])
                    if not v_input.connected:
                        cmd_resp[message.id] = {"success": False, "message": "Failed to add stream."}
                    else:
                        cmd_resp[message.id] = {"success": True, "message": "Successfully connected to stream"}

        elif message.type == "stream_remove":
            logging.debug("Message is a stream remove")
            if v_input.connected:
                if v_input.camera_id != message.data:
                    cmd_resp[message.id] = {"success": False, "message": f"Stream ID {message.data} does not exist. No stream removed."}
                else:
                    v_input.close_stream()
                    cmd_resp[message.id] = {"success": True, "message": "Stream removed successfully"}
            else:
                cmd_resp[message.id] = {"success": False, "message": "No stream connected. Nothing to remove."}

        else:
            raise Exception("Received invalid message type from flask")

    if (image:=v_input()) is not None:

        #resize frame to 1080p for smooth output        
        resized_image = cudaAllocMapped(width=1920, height=1080, format=image.format)
        cudaResize(image, resized_image)

        if objects and thresholds:
            output = predictor.predict(
                image = image,
                text = objects,
                text_encodings = objects_encoding,
                threshold = thresholds,
                pad_square = True
            )

            #Generate overlay and output on the resized image 
            text_labels = [objects[x] for x in output.labels]
            bboxes = output.boxes.tolist()
            resized_bboxes = resize_bboxes(bboxes, (image.width, image.height), (resized_image.width, resized_image.height))

            image = overlay_gen(resized_image, text_labels, resized_bboxes) #generate overlay on resized image

            #Generate metadata in mmj schema and output on redis 
            if frame_counter % config.redis_output_interval == 0:
                schema_gen(text_labels, bboxes) #redis output in original image dimensions
        

        frame_counter+=1
        v_output(resized_image)
