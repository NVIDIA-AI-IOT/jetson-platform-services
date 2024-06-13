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

from time import sleep
from queue import Queue 
import logging 
import os 

#from settings import load_config
from utils import process_alerts, process_query, vlm_alert_handler, vlm_chat_completion_handler
from api_server import VLMServer
from config import load_config
from ws_server import WebSocketServer

from mmj_utils.overlay_gen import VLMOverlay
from mmj_utils.streaming import VideoOutput, VideoSource
from mmj_utils.vlm import VLM 
from mmj_utils.monitoring import AlertMonitor
from jetson_utils import cudaResize, cudaAllocMapped 

#Load config
config_path = os.environ["MAIN_CONFIG_PATH"]
config = load_config(config_path, "main")

logging.basicConfig(level=logging.getLevelName(config.log_level),
                    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

#Setup prometheus states and metric server 
alertMonitor = AlertMonitor(config.max_alerts, config.prometheus_port, cooldown_time=config.alert_cooldown)

#Setup websocket server to send push alerts to mobile app 
ws_server = WebSocketServer(port=config.websocket_server_port)
ws_server.start_server()

#create video input and output using mmj-utils,
v_input = VideoSource() #start as None will connect upon api request
v_output = VideoOutput(config.stream_output)

#create overlay object
overlay = VLMOverlay()

#REST API Queue 
cmd_q = Queue() #commands are put in by REST API requests and taken out by main loop 
cmd_resp = dict() #responses to commands will be put in a shared dict based on command UUID 

#Launch REST API server and connect queue to receive prompt and alert updates 
api_server = VLMServer(cmd_q, cmd_resp, max_alerts=10, port=config.api_server_port)
api_server.start_server()

#Create VLM object
vlm = VLM(config.chat_server)
vlm.add_ego("alert", config.alert_system_prompt, vlm_alert_handler, {"alertMonitor": alertMonitor, "overlay":overlay, "ws_server":ws_server, "v_input":v_input})
vlm.add_ego("chat_completion", callback=vlm_chat_completion_handler, callback_args={"cmd_resp":cmd_resp, "overlay":overlay})

#Wait for VLM server to be loaded 
while not vlm.health_check():
    if not cmd_q.empty():
        message = cmd_q.get()
        cmd_resp[message.id] = "VLM Model is still loading. Please try again later."
    sleep(5)
    logging.info("Waiting for VLM model health check")
logging.info("VLM Model health check passed")

#Start main pipeline 
api_server_check = False #used to ensure api server queue is checked at least once between llm calls 
active_message = None
while True:
    #First check controls from api server and update rules, queries or 
    if not vlm.busy: #LLM is free, accept next command.
        api_server_check = True 
        if not cmd_q.empty():
            message = cmd_q.get() #cmd from api server
            logging.debug("Received message from flask")

            if message.type == "alert":
                logging.debug("Message is an alert")

                alertMonitor.clear_alerts() #clear to stop old rules triggering alerts
                alertMonitor.set_alerts(message.data)
                message.data = process_alerts(message.data)
                overlay.input_text = message.data
                overlay.output_text = None 
                active_message = message 
                cmd_resp[message.id] = "Alert rules set"
     
            elif message.type == "query":
                logging.debug("Message is a query")

                #Clear old alerts
                alertMonitor.clear_alerts() #clear to stop old alerts triggering alerts

                if not v_input.connected:
                    cmd_resp[message.id] = "No stream has been added."
                    active_message = None

                else:
                    overlay.input_text = process_query(message.data)
                    overlay.output_text = None 
                    active_message = message      

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
                        alertMonitor.clear_alerts() #clear to stop old alerts triggering alerts
                else:
                    cmd_resp[message.id] = {"success": False, "message": "No stream connected. Nothing to remove."}

            else:
                raise Exception("Received invalid message type from flask")
        
    #If a stream is added get frame and output 
    if (frame:=v_input()) is not None:
        if not vlm.busy and active_message and api_server_check: #llm available & query or alerts 
            if active_message.type == "alert":
                logging.info("Making VLM call with alert input")
                vlm("alert", active_message.data, frame)
            elif active_message.type == "query":
                logging.info("Making VLM call with query input")
                vlm("chat_completion", active_message.data, [frame], message_id=message.id)
                active_message = None #only send queries 1 time 
            else:
                logging.error(f"Message type is invalid: {active_message.type}")
        
            api_server_check = False   

        #resize frame to 1080p for smooth output        
        resized_frame = cudaAllocMapped(width=1920, height=1080, format=frame.format)
        cudaResize(frame, resized_frame)

        #generate overlay 
        resized_frame = overlay(resized_frame)  
        v_output(resized_frame)

    else:
        sleep(1/30)

