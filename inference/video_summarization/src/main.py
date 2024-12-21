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

from tqdm import tqdm
from config import load_config 
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import base64
from nano_llm import NanoLLM, ChatHistory
import cv2
import requests
from urllib.parse import urlencode
import torch
from pydantic import BaseModel
from typing import Optional, List
from nano_llm.plugins import VideoSource
from jetson_utils import cudaMemcpy
import json
import logging
from datetime import datetime, timedelta
from nano_llm import NanoLLM, ChatHistory
from sklearn.metrics.pairwise import cosine_distances
import redis
import openai
from collections import defaultdict
from datetime import datetime
import os
import numpy as np

class LiveStreamRequest(BaseModel):
    liveStreamUrl: str
    description: Optional[str] = None

class LiveStreamResponse(BaseModel):
    id: str
    url: str
    description: Optional[str] = None

class ModelResponse(BaseModel):
    id: str
    name: str

class VideoSnippet(BaseModel):
    start_time: str
    end_time: str

class StreamSummarizeRequest(BaseModel):
    sensor_id: str
    stream_id: str
    prompt: Optional[str] = None  
    model: str
    snippets: List[VideoSnippet]

class FileSummarizeRequest(BaseModel):
    file_id: str
    prompt: Optional[str] = None  
    model: str
    chunk_duration: Optional[int] = 20

class SummarizeRequest(BaseModel):
    id: Optional[str] = None
    prompt: Optional[str] = None
    stream_id: Optional[str] = None
    model: str
    stream: bool = False
    chunk_duration: Optional[int] = 20
    from_timestamp: Optional[str] = None
    to_timestamp: Optional[str] = None

class ClusteringRequest(BaseModel):
    stream_id: str
    from_timestamp: Optional[str] = None
    to_timestamp: Optional[str] = None

class SummaryClasses(BaseModel):
    sensor_id: str
    from_timestamp: Optional[str] = None
    to_timestamp: Optional[str] = None

def success(message, *args, **kwargs):
    if logging.getLogger().isEnabledFor(logging.INFO):
        logging.log(logging.INFO, f"SUCCESS: {message}", *args, **kwargs)

logging.success = success

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



config_path = os.getenv('CONFIG_PATH', './config/main_config.json')
config = load_config(config_path)

openai.api_key = config.openai_api_key
jetson_ip = config.jetson_ip
video_port = config.video_port
ingress_port = config.ingress_port 
redis_client = redis.Redis(host='localhost', port=config.redis_port, db=0)

model = NanoLLM.from_pretrained(
    model='Efficient-Large-Model/VILA1.5-3b', 
    # api='awq',
    # quantization='/data/models/awq/vila-1.5-3b-w4-g128-awq-v2.pt', 
    vision_api="hf", 
    max_context_len=64, 
    vision_model=None,
    vision_scaling=None, 
)

chat_history = ChatHistory(model) 

sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda', model_kwargs={"torch_dtype": torch.float16})

dbscan_model = DBSCAN(eps=0.3, min_samples=1, metric='precomputed')

def get_video_frames_file(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_input_framerate = int(cap.get(cv2.CAP_PROP_FPS))
    video_duration = total_frames / video_input_framerate if video_input_framerate > 0 else 0
    cap.release()

    num_images = 0
    last_image = None
    frames = []

    def on_video(image):
        nonlocal last_image, num_images, frames
        last_image = cudaMemcpy(image)

        if last_image:
            if num_images % 3 == 0:
                frames.append(last_image)
            num_images += 1

    video_source = VideoSource(video_input=video_path, 
                               num_buffers=30, 
                               cuda_stream=0,
                               video_input_width=1024,
                               video_input_height=768,
                               video_input_framerate=video_input_framerate)
    video_source.add(on_video, threaded=False)
    video_source.start()

    while True:
        if last_image is None:
            continue
        
        if video_source.eos:
            break
    print("FRAMES RETURNED: " + str(len(frames)) + ", VALUES:" + str(frames))
    return frames, video_duration

def summarize_file(body: FileSummarizeRequest):
    file_id, chunk_duration, prompt = body['file_id'], body['chunk_duration'], body['prompt']
    video_path = get_video_path(file_id)
    try:
        sample_frames_list, video_duration = get_video_frames_modified("file://" + str(video_path))

        total_frames = len(sample_frames_list)
        fps = total_frames / video_duration

        summaries = []
        total_chunks = int(video_duration // chunk_duration)
        remaining_time = video_duration % chunk_duration

        for chunk_index in range(total_chunks + (1 if remaining_time > 0 else 0)): 
            start_frame_idx = int(chunk_index * chunk_duration * fps)
            end_frame_idx = int((chunk_index + 1) * chunk_duration * fps)

            frames = sample_frames_list[start_frame_idx:end_frame_idx]
            
            start_time = chunk_index * chunk_duration
            end_time = start_time + len(frames) / fps  
            
            timestamp = f"{int(start_time // 60):02}:{int(start_time % 60):02}-{int(end_time // 60):02}:{int(end_time % 60):02}"
            
            summary = _summarize(
                body["model"],
                frames,
                prompt="Examine these video frames and provide a detailed, accurate description of observable actions. Focus exclusively on visible activity without making inferences or assumptions. Remember, these frames capture the same person over time, not different individuals. Be concise and precise in your description." if not prompt else prompt 
            )

            summaries.append(f"{timestamp}: {summary}")

        final_summary = caption_integration(summaries, 
                            "Provide a high-level summarization of the entire video, focusing on overall activity. Use the provided summaries of continuous video chunks to inform your summary. Do not analyze each scene individually.")
        print('summaries:', summaries)
        print('final_summary:', final_summary)
        
        log_entry = {
            "summaries": summaries,
            "final_summary": final_summary
        }
        
        dump_summarization(file_id, log_entry)

        return final_summary

    except Exception as e:
        print(f"An error occurred in summarize_file: {e}")

def get_video_frames(stream_id, start_time, end_time):
    url = f"http://{config.jetson_ip}:{config.video_port}/api/v1/storage/file/{stream_id}"
    params = {
        'startTime': start_time,
        'endTime': end_time
    }
    url = f"{base_url}?{urlencode(params)}"
    fames, duration = get_video_frames_modified(url)
    return frames

def get_video_frames_modified(url):
    print(f"Capturing Videos from: {url}")

    try:
        camSet = f'uridecodebin uri={url} ! queue ! nvvideoconvert compute-hw=1 ! video/x-raw,format=BGR ! queue ! appsink'
        cap = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print("Error: Unable to open video stream.")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) # 30
        frame_interval = fps // 3 # int(fps)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        print(f"Extracted {len(frames)} frames from the video stream.")
        cap2 = cv2.VideoCapture(url)
        if not cap2.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return []

        total_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        video_input_framerate = int(cap2.get(cv2.CAP_PROP_FPS))
        video_duration = total_frames / video_input_framerate if video_input_framerate > 0 else 0
        cap2.release()

        print("total_frames: " + str(total_frames) + ", video_input_framerate: " + str(video_input_framerate) + ", video_duration: " + str(video_duration))
        video_frames = np.stack(frames, axis=0)
        return video_frames, video_duration 
    
    except Exception as e:
        print(f"An error occurred in get_video_frames_modified: {e}")
        return None, None

def caption_integration(text, prompt, max_tokens=512):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n{text}"}
        ],
        max_tokens=max_tokens,
        temperature=0.5,
    )
    return response['choices'][0]['message']['content'].strip()

def summarize_vila(prompt, frames):
    outputs = {}
    batch_size = 8

    for i in tqdm(range(0, len(frames), batch_size)):
        batch_frames = frames[i:i + batch_size]
        chat_history.reset()  
        chat_history.append('user', prompt)

        for idx, image in enumerate(batch_frames):
            chat_history.append('user', text=f'Image {idx}:')
            chat_history.append('user', image=image)

        embedding, _ = chat_history.embed_chat()

        output = model.generate(
            embedding, 
            streaming=False,
            kv_cache=chat_history.kv_cache,
            stop_tokens=chat_history.template.stop,
            max_new_tokens=128,
            min_new_tokens=-1,
            do_sample=False,
            repetition_penalty=1.0,
            temperature=0.7,
            top_p=0.95,
        )

        outputs[i] = output.strip()[:-4]

    return outputs

def summarize_gpt4(prompt, frames):
    def encode_image(image_array):
        _, buffer = cv2.imencode('.jpg', image_array)
        return base64.b64encode(buffer).decode('utf-8')

    outputs = {}

    num_images_to_load = config.max_images
    num_chunks = (len(frames) + num_images_to_load - 1) // num_images_to_load 

    for idx in range(num_chunks):
        start_idx = idx * num_images_to_load
        end_idx = min(start_idx + num_images_to_load, len(frames))
        
        selected_frames = frames[start_idx:end_idx]
        base64_images = [encode_image(frame) for frame in selected_frames]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}} for image in base64_images
                ]
            }
        ]
        
        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 1024
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        
        outputs[idx] = response_json['choices'][0]['message']['content']

    return outputs

def merge_captions(labels):
    n = len(labels)
    indicator = [True] * n

    for i in range(n):
        if labels[i] == -1:
            indicator[i] = False

    for i in range(n - 4):
        window = labels[i:i + 5]
        most_common_label = max(set(window), key=window.count)
        
        if window.count(most_common_label) == 4:
            for j in range(5):
                if window[j] != most_common_label:
                    indicator[i + j] = False
                    break

    return indicator

def _clustering(captions): # at the level of summarization
    keys = list(captions.keys())
    values = list(captions.values())

    embeddings = sentence_model.encode(values, convert_to_tensor=True).cpu().numpy()

    distance_matrix = cosine_distances(embeddings)

    labels = dbscan_model.fit_predict(distance_matrix)

    clusters = defaultdict(dict)

    for idx, label in enumerate(labels):
        clusters[label][keys[idx]] = values[idx]

    return clusters

def clustering_denoise(captions): # at the level of captions
    embeddings = sentence_model.encode(list(captions.values()), convert_to_tensor=True).cpu().numpy()

    distance_matrix = cosine_distances(embeddings)

    labels = dbscan_model.fit_predict(distance_matrix)

    indicator = merge_captions(labels.tolist())

    filtered_captions = {key: value for key, value, select in zip(captions.keys(), captions.values(), indicator) if select}

    return filtered_captions

def _summarize(model, frames, prompt):
    outputs = {}
    if model == 'vila-1.5':
        outputs = summarize_vila(prompt, frames)
    elif model == 'gpt-4o':
        outputs = summarize_gpt4(prompt, frames)
    else:
        raise ValueError(f"Unsupported model: {model}")

    outputs = clustering_denoise(outputs)
    captions = ", ".join(f"{key}: '{value}'" for key, value in outputs.items())
    summary = caption_integration(captions, 
                            "Analyze the frame-by-frame descriptions of human activities below. Summarize the continuous activity into a single, cohesive sentence starting with the subject. Captions may contain errors, so ignore captions that are less frequently mentioned in the overall description. Avoid generating unreal or unsure interactions by disregarding any irrelevant information about human activity.")
    print(captions)
    print(summary)

    return summary

def dump_summarization(key, log_entry):
    redis_client.rpush(key, json.dumps(log_entry))

def dump_summarization_clustered(stream_id, summarization_clustered):
    sensor_id = get_sensor_id(stream_id)
    for time_range, summary_data in summarization_clustered.items():
        summaries = summary_data['summaries']
        representative_text = summary_data['representative_text']
        representative_id = f"rep_{hash(representative_text)}"

        for summary_time_range, summary in summaries.items():
            date_key = datetime.strptime(summary_time_range[0], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
            redis_key = f"summary_logs_{sensor_id}:{date_key}"
            log_entry = {
                "start_time": datetime.strptime(summary_time_range[0], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.strptime(summary_time_range[1], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'),
                "summary": summary,
            }
            redis_client.lrem(redis_key, 1, json.dumps(log_entry))
            log_entry["representative_id"] =  representative_id
            redis_client.rpush(redis_key, json.dumps(log_entry))

        log_entry_representative = {
            "start_time": datetime.strptime(time_range[0], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": datetime.strptime(time_range[1], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'),
            "summary": representative_text,
            "id": representative_id
        }
        redis_client.rpush(redis_key, json.dumps(log_entry_representative))
    dump_digest(sensor_id)

def dump_digest(sensor_id):
    yesterday_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    daily_report_key = f"daily_report_{sensor_id}:{yesterday_date}"
    if not redis_client.exists(daily_report_key):
        yesterday_log_key = f"summary_logs_{sensor_id}:{yesterday_date}"
        
        if redis_client.exists(yesterday_log_key):
            yesterday_logs = redis_client.lrange(yesterday_log_key, 0, -1)
            yesterday_summaries = "\n".join(
                f"[{json.loads(log)['start_time']}, {json.loads(log)['end_time']}] {json.loads(log)['summary']}"
                for log in yesterday_logs if 'id' in json.loads(log)
            )
            # summarized_text = caption_integration(yesterday_summaries, "Below are summaries of human activities throughout the day. Please give a detailed daily digest for that, and also a high-level digest", max_tokens=1024)
            summarized_text = caption_integration(yesterday_summaries, """
                Below are summaries of human activities throughout the day. Please provide a detailed daily digest in the following format:

                - [Time] [Subject] [Action]

                For example:
                - 9:00 AM Children went out
                - 10:45 AM Person watered the lawn
                - 1:00 PM Mail man delivered the mail
                - 3:00 PM Dog was playing in the front lawn

                Please include as many events as possible, arranged chronologically. Use precise times when available, or approximate times if not specified. Ensure each entry follows the format: time, subject, and action.

                Additionally, provide a high-level summary of the day's activities in a brief paragraph.
                """, max_tokens=1024)
            
            redis_client.set(daily_report_key, summarized_text)
            print(f"Generated and saved daily report for {yesterday_date}")

def fetch_summarization(stream_id, start_datetime, end_datetime):
    logs = []
    sensor_id = get_sensor_id(stream_id)
    current_date = start_datetime.date()
    end_date = end_datetime.date()

    while current_date <= end_date:
        log_key = f"summary_logs_{sensor_id}:{current_date}"
        day_logs = redis_client.lrange(log_key, 0, -1)
        
        for log in day_logs:
            log_entry = json.loads(log)
            log_start = datetime.strptime(log_entry['start_time'], '%Y-%m-%d %H:%M:%S')
            log_end = datetime.strptime(log_entry['end_time'], '%Y-%m-%d %H:%M:%S')
            
            if start_datetime <= log_start and log_end <= end_datetime:
                logs.append(log_entry)

        current_date += timedelta(days=1)
    return logs

def clustering_summarization(clusters):
    summarization_clustered = {}
    for cluster in clusters.values():
        timestamps = list(cluster.keys())
        summary = list(cluster.values())
        summary_clustered = caption_integration(summary, "The following are summaries of related human activities grouped together. Please provide a representative description that captures the essence of this cluster of activities. Describe the activity directly starting with the subject.", 
                                                max_tokens=256) if len(cluster) != 1 else summary[0]
        summarization_clustered[(timestamps[0][0], timestamps[-1][1])] = {
            'representative_text': summary_clustered,
            'summaries': cluster
        }
    return summarization_clustered

def clustering(request: ClusteringRequest):
    print(request.dict())
    if isinstance(request, dict):
        stream_id, from_timestamp, to_timestamp = request['stream_id'], request['from_timestamp'], request['to_timestamp']
    else:
        stream_id, from_timestamp, to_timestamp = request.stream_id, request.from_timestamp, request.to_timestamp

    try:
        start_datetime = datetime.strptime(from_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
        end_datetime = datetime.strptime(to_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ') if to_timestamp else datetime.utcnow()
    except ValueError:
        return {"error": "Invalid datetime format. Use ISO format: YYYY-MM-DDTHH:MM:SS.sssZ"}
    
    # fetch summarization
    summarization = fetch_summarization(stream_id, start_datetime, end_datetime)

    # clustering 
    summarization_dict = {(item['start_time'], item['end_time']): item['summary'] for item in summarization if 'id' not in item and 'representative_id' not in item}
    
    if not summarization_dict:
        print(f'Summarization list is empty for {stream_id} from {from_timestamp} to {to_timestamp}')
        return
    
    clusters = _clustering(summarization_dict)
    
    # merge summarization
    summarization_clustered = clustering_summarization(clusters)

    # store clustered summarization
    dump_summarization_clustered(stream_id, summarization_clustered)

def get_video_path(file_id):
    try:
        file_info_json = redis_client.hget("files", file_id)
        file_info = json.loads(file_info_json.decode('utf-8'))
        return file_info['filepath']
    except Exception as e:
        print(f"An unexpected error occurred during getting video path: {e}")
        return None

def summarize_stream(body: StreamSummarizeRequest):
    prompt, sensor_id = body["prompt"], body["sensor_id"]

    for snippet in body["snippets"]:
        images = get_video_frames(body["stream_id"], snippet['start_time'], snippet['end_time'])
        
        if len(images) < 6:
            continue
        
        summary = _summarize(
            body["model"],
            images,
            prompt = "Examine the human in these video frames and provide a detailed, accurate description of their observable actions. Focus exclusively on visible activity without making inferences or assumptions. Remember, these frames capture the same person over time, not different individuals. Be concise and precise in your description." if not prompt else prompt
        )

        log_entry = {
            "start_time": datetime.strptime(snippet["start_time"], '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": datetime.strptime(snippet["end_time"], '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d %H:%M:%S'),
            "summary": summary
        }
        
        today_date = datetime.today().strftime('%Y-%m-%d')
        key = f"summary_logs_{sensor_id}:{today_date}"
        
        dump_summarization(key, log_entry)

def get_sensor_id(stream_id):
    url = f'http://{jetson_ip}:81/api/v1/live/streams'
    headers = {
        'accept': 'application/json'
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        streams = response.json()
        for entry in streams:
            for _, stream_list in entry.items():
                for stream in stream_list:
                    if stream['streamId'] == stream_id:
                        return stream['name']
        print(f"Failed to retrieve sensor id with stream_id {stream_id}")
    else:
        print(f"Failed to retrieve streams data, status code: {response.status_code}")

def get_stream_id_from_sensor(sensorId):
    url = f'http://{jetson_ip}:81/api/v1/live/streams'
    headers = {
        'accept': 'application/json'
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        streams = response.json()
        for entry in streams:
            for sensor_id, stream_list in entry.items():
                if sensor_id == sensorId:
                    return stream_list[0]['streamId']
        print(f"Failed to retrieve sensor id with sensorId {sensorId}")
    else:
        print(f"Failed to retrieve streams data, status code: {response.status_code}")

def get_stream_id(stream_url):
    url = f'http://{jetson_ip}:81/api/v1/live/streams'
    headers = {
        'accept': 'application/json'
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        streams = response.json()
        for entry in streams:
            for _, stream_list in entry.items():
                for stream in stream_list:
                    if stream['url'] == stream_url:
                        return stream['streamId']
    # no found in vst
    return None

def get_file_content(file_id: str):
    url = f"https://localhost/api/file/metadata?fileId={file_id}"
    headers = {'accept': 'application/json'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses

        return {"file_id": file_id, "content": response.json()}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve file content: {e}", "file_id": file_id}

def add_to_vst(stream_url):
    try:
        streamname = stream_url.split('/')[-1]
        url = f"http://{jetson_ip}:81/api/v1/sensor/add"
        headers = {"Content-Type": "application/json"}

        payload = {
            "sensorUrl": stream_url,
            "name": streamname
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))

        response.raise_for_status()  

        response_data = response.json()

        sensor_id = response_data['sensorId']
        
        return sensor_id
    
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None

def parse_timestamp(timestamp):
    if timestamp:
        return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
    
    return None

def get_fov_histogram(request: SummaryClasses, current_timestamp):
    stream_id = request.stream_id
    from_timestamp = current_timestamp - timedelta(seconds=20)
    to_timestamp = min(current_timestamp, parse_timestamp(request.to_timestamp)) if request.to_timestamp else current_timestamp

    base_url = f"http://{jetson_ip}:{ingress_port}/emdx"
    endpoint = "/api/metrics/occupancy/fov/histogram"
    
    from_timestamp_str = from_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    to_timestamp_str = to_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    sensor_id = get_sensor_id(stream_id)
    url = f"{base_url}{endpoint}?sensorId={sensor_id}&fromTimestamp={from_timestamp_str}&toTimestamp={to_timestamp_str}"
    print(url)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return sensor_id, response.json()
        else:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)
            return sensor_id, None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred in get_fov_histogram: {e}")
        return sensor_id, None

def extract_video_snippet(fov):
    if fov and 'counts' in fov and len(fov['counts']) > 0:
        histogram = fov['counts'][0].get('histogram', [])
        snippets = []
        current_snippet = None
        
        for entry in histogram:
            if entry['max_count'] > 0:
                start_time = datetime.strptime(entry['start'], "%Y-%m-%dT%H:%M:%S.%fZ")
                end_time = datetime.strptime(entry['end'], "%Y-%m-%dT%H:%M:%S.%fZ")
                
                if current_snippet is None:
                    current_snippet = {'start': start_time, 'end': end_time}
                elif (start_time - current_snippet['end']) <= timedelta(seconds=5):
                    current_snippet['end'] = end_time
                else:
                    snippets.append(current_snippet)
                    current_snippet = {'start': start_time, 'end': end_time}
        
        if current_snippet:
            snippets.append(current_snippet)
        
        return [
            (snippet['start'].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            snippet['end'].strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            for snippet in snippets
        ]
    else:
        print("The response does not contain the expected 'counts' and 'histogram' structure.")
        return []
