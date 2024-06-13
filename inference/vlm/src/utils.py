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

import logging 
import json
import re 

#process alerts
def process_alerts(alerts):
    
    if alerts.get("r0", None) == "No rule":
        alert_str = "No alert set. Briefly describe the scene."
    
    elif len(alerts) == 0:
        alert_str = "No alert set. Briefly describe the scene."

    else:
        alert_str = str(alerts)
        alert_str = f"Determine if each alert is True or False: {alert_str}. Respond in JSON format."

    return str(alert_str)

#process alerts reply from llm
def process_alerts_reply(reply):
    try:
        reply = reply.replace("</s>", "")
        reply = reply.replace("\n", "")
        reply = reply.strip()

        reply = reply.lower()
        reply = reply.replace("no", "False")
        reply = reply.replace("yes", "True")
        reply = reply.replace("false", "False")
        reply = reply.replace("true", "True")

        reply = reply.replace("'", '"') #convert single quotes to double quotes

        #ensure quotes around False 
        pattern = r'\b(?<!")' + re.escape("True") + r'(?!")\b'
        reply = re.sub(pattern, f'"True"', reply)

        #ensure quotes around True 
        pattern = r'\b(?<!")' + re.escape("False") + r'(?!")\b'
        reply = re.sub(pattern, f'"False"', reply)

        #remove anything not in brackets  
        if "{" in reply:
            start_index = reply.index("{")
        else:
            start_index = 0

        if "}" in reply:
            end_index = reply.index("}")
        else:
            end_index = len(reply)-1
        
        reply = reply[start_index:end_index+1]

        #ensure brackets 
        if reply[0] != "{":
            reply = "{" + reply 
        if reply[-1] != "}":
            reply = reply + "}"

        #convert to json 
        reply = json.loads(reply)

        #convert true/false to 1/0
        for key in reply:
            reply[key] = 1 if reply[key] == "True" else 0 

    except Exception as e:
        logging.debug(e)
        logging.info(f"LLM Reply was not in JSON format: {reply}")
        reply = dict()

    return reply 


def process_query(query):
    for message in query.messages:
        if message.role == "user":
            if isinstance(message.content, str):
                return message.content 
            #if not string should be list 
            for content in message.content:
                if content.type == "text":
                    return content.text 
    return "No user prompt found."



def vlm_alert_handler(response, **kwargs):
    try:
        logging.info("Updating prometheus alerts")
        v_input = kwargs["v_input"]
        
        #when alert states change, update prometheus and overlay
        alertMonitor = kwargs["alertMonitor"]
        alert_states = process_alerts_reply(response)
        logging.info(f"Updated alert states: {alert_states}")
        

        alert_cooldowns = {key: alertMonitor.alerts[key].cooldown for key in alertMonitor.alerts}
        alertMonitor.set_alert_states(alert_states)

        ws_server = kwargs["ws_server"]

        for alert_id, alert in alertMonitor.alerts.items():
            logging.info(alert_id)
            alert_string = alert.string 
            alert_state = alert.state
            logging.info(alert_state)
            if alert_state == 1 and not alert_cooldowns[alert_id]:
                data = {"stream_url":v_input.url, "stream_id":v_input.camera_id, "camera_name":v_input.camera_name, "alert_id":alert_id, "alert_str":alert_string}
                logging.info(data)
                ws_server.send_alert(data)
        
        logging.info("updating overlay")
        overlay = kwargs["overlay"]
        overlay.output_text = response
        overlay.reset_decay()
    except Exception as e:
        logging.info(e)

def vlm_chat_completion_handler(response, **kwargs):
    overlay = kwargs["overlay"]
    cmd_resp = kwargs["cmd_resp"]
    message_id = kwargs["message_id"]
    logging.info("Sending query response")
    cmd_resp[message_id] = response
    overlay.output_text = response["choices"][0]["message"]["content"]
    overlay.reset_decay()
