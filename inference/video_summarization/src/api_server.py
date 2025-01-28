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

import time
from mmj_utils.api_server import APIServer, APIMessage
from fastapi import FastAPI, HTTPException
import asyncio
from typing import Optional
from uuid import uuid4
from queue import PriorityQueue
from mmj_utils.api_server import APIServer
import redis
from typing import Optional
import uuid
from datetime import datetime, timedelta
from config import load_config 
import json
from main import get_fov_histogram, extract_video_snippet, summarize_file, summarize_stream, parse_timestamp, clustering, get_sensor_id, fetch_summarization, get_stream_id, add_to_vst, get_stream_id_from_sensor
from main import StreamSummarizeRequest, FileSummarizeRequest, VideoSnippet, LiveStreamRequest, LiveStreamResponse, ModelResponse, SummarizeRequest
import os

class VideoSummaryServer(APIServer):

    def __init__(self, cmd_q, resp_d, port=19000, clean_up_time=180):
        super().__init__(cmd_q, resp_d, port=port, clean_up_time=clean_up_time)

        config_path = os.getenv('CONFIG_PATH', './config/main_config.json')
        config = load_config(config_path)

        self.jetson_ip = config.jetson_ip
        self.video_port = config.video_port
        self.streamer_port = config.streamer_port
        self.ingress_port = config.ingress_port 
        self.segment_time = config.segment_time 
        self.redis_client = redis.Redis(host=config.jetson_ip, port=config.redis_port, db=0)
        self.live = False
        self.ready = False
        self.models = []
        
        self.app = FastAPI()

        self.app.post("/files")(self.upload_file)
        self.app.get("/files")(self.list_files)
        self.app.delete("/files/{file_id}")(self.delete_file)
        self.app.get("/files/{file_id}")(self.get_file)

        self.app.post("/live-stream")(self.add_live_stream)
        self.app.get("/live-stream")(self.list_live_streams)
        self.app.delete("/live-stream/{stream_id}")(self.remove_live_stream)

        self.app.get("/health/ready")(self.health_ready)
        self.app.get("/health/live")(self.health_live)

        self.app.get("/models")(self.list_models)
        
        self.app.post("/summarize")(self.run_video_summarize_query)
        self.app.get("/summarize")(self.fetch_summarization)
        self.app.get("/summarize/digest")(self.fetch_digest)
        
        self.app.post("/clustering")(clustering)

    # Video Files
    async def upload_file(self, filepath: str):
        file_id = str(uuid.uuid4())  
        file_data = {
            'filepath': filepath,
        }
        self.redis_client.hset('files', file_id, json.dumps(file_data))

        return file_id

    async def list_files(self):
        files = self.redis_client.hgetall('files')
        
        file_list = {}
        for file_id, data in files.items():
            file_info = json.loads(data.decode('utf-8'))
            file_list[file_id.decode('utf-8')] = file_info
        
        return file_list

    async def delete_file(self, file_id: str):
        if self.redis_client.hexists('files', file_id):
            self.redis_client.hdel('files', file_id)
            return {"status": "success", "message": f"File with ID {file_id} deleted."}
        else:
            return {"status": "error", "message": f"File with ID {file_id} not found."}

    async def get_file(self, file_id: str):
        if self.redis_client.hexists("files", file_id):
            file_info_json = self.redis_client.hget("files", file_id)
            file_info = json.loads(file_info_json.decode('utf-8'))
            
            return {
                "file_id": file_id,
                "filepath": file_info.get('filepath'),
            }
        else:
            return {"error": "File ID not found", "file_id": file_id}

    # Live Streams
    async def add_live_stream(self, request: LiveStreamRequest):
        try:
            stream_url = request.liveStreamUrl
            stream_id = get_stream_id(stream_url)

            if not stream_id:
                sensor_id = add_to_vst(stream_url)
                stream_id = get_stream_id_from_sensor(sensor_id)

            stream_data = {
                "url": stream_url,
                "id": stream_id,
                "description": request.description,
            }
            self.redis_client.hset("live_streams", stream_id, json.dumps(stream_data))
            return LiveStreamResponse(**stream_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    async def list_live_streams(self):
        try:
            live_streams = []
            keys = self.redis_client.hkeys("live_streams")
            for key in keys:
                stream_data = json.loads(self.redis_client.hget("live_streams", key))
                live_streams.append(LiveStreamResponse(**stream_data))
            return live_streams
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    async def remove_live_stream(self, stream_id: str):
        try:
            if self.redis_client.hexists("live_streams", stream_id):
                self.redis_client.hdel("live_streams", stream_id)
                return {"message": "Operation Successful"}
            else:
                raise HTTPException(status_code=422, detail="Stream ID not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    # Microservice Health
    async def health_ready(self):
        try:
            if self.ready:
                return {"status": "ready"}
            else:
                raise HTTPException(status_code=503, detail="Service not ready")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    async def health_live(self):
        try:
            if self.live:
                return {"status": "live"}
            else:
                raise HTTPException(status_code=503, detail="Service not live")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    def ready_set(self):
        self.ready = True

    def live_set(self):
        self.live = True

    # Model 
    async def list_models(self):
        try:
            return [ModelResponse(**model) for model in self.models]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def model_register(self):
        self.models = [
            {
                "id": str(uuid.uuid4()),
                "name": "vila-1.5"
            },
            {
                "id": str(uuid.uuid4()),
                "name": "gpt-4o"
            }
        ]
        
    # Summarization with AI-NVR: POST and GET 
    async def run_video_summarize_query(self, request: SummarizeRequest):
        # Files 
        if not request.stream:
            request_data = FileSummarizeRequest(file_id=request.id, prompt=request.prompt, model=request.model, chunk_duration=request.chunk_duration)
            queue_message = APIMessage(data=request_data.dict(), type="process_file", id=str(uuid4()), time=time.time())
            start_time = datetime.utcnow().timestamp()
            self.cmd_q.put(((start_time, request.id), queue_message))
            self.cmd_tracker[queue_message.id] = queue_message.time
            return 

        # Stream
        from_timestamp = parse_timestamp(request.from_timestamp)
        to_timestamp = parse_timestamp(request.to_timestamp)

        while True:
            if from_timestamp:
                current_timestamp = from_timestamp + timedelta(seconds=20)
                from_timestamp = current_timestamp
            else:
                current_timestamp = datetime.utcnow() + timedelta(seconds=20)
           
            current_timestamp_seconds = time.mktime(current_timestamp.timetuple()) + current_timestamp.microsecond / 1e6
            realtime = time.time()
            if current_timestamp_seconds > realtime:
                await asyncio.sleep(current_timestamp_seconds - realtime)
            
            try:
                sensor_id, fov = get_fov_histogram(request, current_timestamp)
            except Exception as e:
                print(f"Error getting FOV histogram: {e}")
                continue

            try:
                snippets = extract_video_snippet(fov)
            except Exception as e:
                print(f"Error extracting video snippet: {e}")
                continue

            if snippets:
                request_data = StreamSummarizeRequest(sensor_id=sensor_id, stream_id=request.stream_id, model=request.model, snippets=[VideoSnippet(start_time=start, end_time=end) for start, end in snippets])
                queue_message = APIMessage(data=request_data.dict(), type="process_stream", id=str(uuid4()), time=time.time())
                start_time = datetime.strptime(request_data.snippets[0].start_time, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
                self.cmd_q.put(((start_time, request.stream_id), queue_message))
                self.cmd_tracker[queue_message.id] = queue_message.time

            if to_timestamp and current_timestamp >= to_timestamp:
                break

    async def fetch_summarization(
            self, 
            stream: Optional[bool] = False, 
            id: Optional[str] = None, 
            stream_id: Optional[str] = None, 
            from_timestamp: Optional[str] = None,
            to_timestamp: Optional[str] = None
        ):    
        if not stream:
            logs = self.redis_client.lrange(id, 0, -1) 
            if not logs:
                return {"message": "No logs found for the given id"}

            log_entries = [json.loads(log.decode('utf-8')) for log in logs]
            return {"file_id": id, "summaries": log_entries[0]['summaries'], "final_summary": log_entries[0]["final_summary"]}
        try:
            start_datetime = datetime.strptime(from_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
            end_datetime = datetime.strptime(to_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ') if to_timestamp else datetime.utcnow()
        except ValueError:
            return {"error": "Invalid datetime format. Use ISO format: YYYY-MM-DDTHH:MM:SS.sssZ"}

        logs = fetch_summarization(stream_id, start_datetime, end_datetime)
        
        if logs:
            return {"stream": stream_id, "logs": logs}
        else:
            return {"error": f"No logs found between {from_timestamp} and {to_timestamp}"}

    async def fetch_digest(self, stream_id: str, date: str):
        sensor_id = get_sensor_id(stream_id)

        daily_report_key = f"daily_report_{sensor_id}:{date}"
        report = self.redis_client.get(daily_report_key)
        if not report:
            return {"error": f"No daily report found for date {date}"}

        return {"sensor": sensor_id, "stream": stream_id, "daily_report": report.decode('utf-8')}


if __name__ == '__main__':
    cmd_q = PriorityQueue()
    cmd_resp = dict()

    config_path = os.getenv('CONFIG_PATH', './config/main_config.json')
    config = load_config(config_path)

    api_server = VideoSummaryServer(cmd_q, cmd_resp, port=config.api_server_port)
    api_server.start_server()
    api_server.model_register()
    api_server.live_set()
    api_server.ready_set()

    while True:
        if not cmd_q.empty():
            print('Size of queue: ', cmd_q.qsize())
            _, message = cmd_q.get()
            print(message.data)

            try:
                if message.type == 'process_stream':
                    summarize_stream(message.data)
                    cmd_resp[message.id] = {"success": True, "message": "Processing successfully."}
                elif message.type == 'process_file':
                    summarize_file(message.data)
                    cmd_resp[message.id] = {"success": True, "message": "Processing successfully."}
                else:
                    print(f"Received invalid message type: {message.type}")
                    cmd_resp[message.id] = {"success": False, "message": f"Received invalid message type: {message.type}"}
            except Exception as e:
                print(f"An error occurred while processing message {message.id}: {e}")
                cmd_resp[message.id] = {"success": False, "message": str(e)}