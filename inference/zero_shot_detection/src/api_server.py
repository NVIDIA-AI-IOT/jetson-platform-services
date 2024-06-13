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

#Setup endpoint that can be used to update the prompt 
from uuid import uuid4
from time import sleep, time 
from mmj_utils.api_server import APIServer, APIMessage, Response 
from pydantic import BaseModel, conlist, constr, confloat 
from typing import List, Optional 
from fastapi import HTTPException 

class DetectionClasses(BaseModel):
    objects: conlist(constr(max_length=100), min_length=0, max_length=25)
    thresholds: conlist(confloat(ge=0.0, le=100.0), min_length=0, max_length=25)
    id: Optional[constr(max_length=100)] = ""

class DetectionServer(APIServer):

    def __init__(self, cmd_q, resp_d, port=5000, clean_up_time=180):
        super().__init__(cmd_q, resp_d, port=port, clean_up_time=clean_up_time)
        
        self.app.post("/api/v1/detection/classes")(self.detection_classes)

    def detection_classes(self, body:DetectionClasses):
        
        queue_message = APIMessage(data=body.dict(), type="classes", id=str(uuid4), time=time())
        self.cmd_q.put(queue_message)
        self.cmd_tracker[queue_message.id] = queue_message.time

        response = self.get_command_response(queue_message.id)
        if response:
            return Response(detail=response)
        else:
            raise HTTPException(status_code=500, detail="Server timed out processing the request")