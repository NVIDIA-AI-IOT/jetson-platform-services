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

import gradio as gr
import requests
from datetime import datetime
from gradio_calendar import Calendar
from config import load_config 
import os

config_path = os.getenv('CONFIG_PATH', './config/main_config.json')
config = load_config(config_path)

jetson_ip = config.jetson_ip
video_port = config.video_port

def fetch_sensor_names():
    try:
        url = f'http://{jetson_ip}:81/api/v1/live/streams'
        headers = {'accept': 'application/json'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        streams = response.json()
        sensor_names = []
        for stream_data in streams:
            for key, stream in stream_data.items():
                if not stream:
                    continue
                sensor_names.append(stream[0]['name'])
        return sensor_names
    except Exception as e:
        return [f"Error fetching sensor names: {str(e)}"]

sensor_names = fetch_sensor_names()

def format_summary(summary):
    logs = summary['logs']
    
    # Sort logs by start_time
    logs.sort(key=lambda x: datetime.fromisoformat(x['start_time'].replace("Z", "+00:00")))
    
    # Create a dictionary to hold representative summaries and their associated summaries
    rep_summaries = {}
    
    for log in logs:
        if 'id' in log:
            rep_id = log['id']
            rep_summaries[rep_id] = {
                'representative_summary': f"[{log['start_time']} - {log['end_time']}]\nEvent: {log['summary']}\n",
                'summaries': []
            }

    for log in logs:
        if 'representative_id' in log:
            rep_id = log['representative_id']
            if rep_id in rep_summaries:
                rep_summaries[rep_id]['summaries'].append(f"|--- [{log['start_time']} - {log['end_time']}] {log['summary']}\n")
    
    # Format the final summary output
    formatted_summary = ""
    for rep_id, data in rep_summaries.items():
        formatted_summary += data['representative_summary']
        for summary in data['summaries']:
            formatted_summary += summary
        formatted_summary += "\n"  # Add a newline after each representative summary block
    
    return formatted_summary.strip() 

def load_video_summary(start_date, start_hour, start_minute, start_second, end_date, end_hour, end_minute, end_second, sensor_id):
    try:
        stream_id = get_stream_id(sensor_id)
        start_time = datetime.combine(start_date, datetime.min.time()).replace(hour=int(start_hour), minute=int(start_minute), second=int(start_second))
        end_time = datetime.combine(end_date, datetime.min.time()).replace(hour=int(end_hour), minute=int(end_minute), second=int(end_second))
        
        if end_time < start_time:
            raise ValueError("End date and time must be after start date and time.")
        
        from_timestamp = start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        to_timestamp = end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        # Prepare API request
        url = "http://localhost:19000/summarize"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        params = {
            "stream": True,
            "stream_id": stream_id,
            "from_timestamp": from_timestamp,
            "to_timestamp": to_timestamp
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        summary = response.json()

        formatted_summary = format_summary(summary)
        
        return formatted_summary
    
    except Exception as e:
        return f"Error: {str(e)}"

def get_stream_id(sensor_id):
    url = f"http://{jetson_ip}:{video_port}/api/v1/live/streams"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            streams = response.json()
            for stream in streams:
                for stream_id, stream_details in stream.items():
                    for detail in stream_details:
                        if detail['name'] == sensor_id:
                            return stream_id
            print(f"No stream found with the name: {sensor_id}")
            return None
        else:
            print(f"Error getting streams: Status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while getting streams: {e}")
        return None
    
def load_daily_digest(digest_date, sensor_id):
    try:
        stream_id = get_stream_id(sensor_id)
        date_str = digest_date.strftime("%Y-%m-%d")
        
        url = "http://localhost:19000/summarize/digest"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        params = {
            "stream_id": stream_id,
            "date": date_str
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        digest = response.json()
        
        formatted_digest = f"{date_str} "
        formatted_digest += f"{digest['daily_report']}\n"
        
        return formatted_digest
    
    except Exception as e:
        return f"Error: {str(e)}"

def fetch_summary_via_http(file_id):
    try:
        url = f"http://localhost:19000/summarize?stream=false&id={file_id}"
        response = requests.get(url)
        response.raise_for_status()
        summary = response.json()
        
        if 'summaries' not in summary or 'final_summary' not in summary:
            return "Summarization is still being generated."
        
        summaries = summary.get('summaries', [])
        final_summary = summary.get('final_summary', "")
        
        result = '\n'.join(summaries) + '\n\nSummary:\n' + final_summary
        return result
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching summary: {str(e)}"
    
def fetch_all_files():
    try:
        url = "http://localhost:19000/files"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        files = response.json()
        file_ids = list(files.keys())
        file_paths = [file['filepath'] for file in files.values()]  # Assuming each file has a 'filepath' field
        path_to_id = {file['filepath']: file_id for file_id, file in files.items()}  # Create a mapping of path to id
        return file_paths, path_to_id
    except Exception as e:
        return [f"Error fetching files: {str(e)}"], {}

def update_summary(selected_file_path, path_to_id):
    file_id = path_to_id.get(selected_file_path, None)
    if file_id:
        return fetch_summary_via_http(file_id)
    else:
        return "File ID not found for the selected path."

file_paths, path_to_id = fetch_all_files() 

with gr.Blocks() as demo:
    gr.Markdown("# Video Summarization Microservice")
    
    gr.Markdown("# File Summarization")
    gr.Markdown("Select a file to fetch its summarization.")

    file_list = gr.Dropdown(label="Files", choices=file_paths)  
    file_summary_output = gr.Textbox(label="File Summary", lines=10)

    file_list.change(fn=lambda x: update_summary(x, path_to_id), inputs=file_list, outputs=file_summary_output)

    gr.Markdown("# Stream Summarization")
    gr.Markdown("Enter the start and end times to fetch the video summary of the camera.")

    with gr.Row():
        start_date = Calendar(type="datetime", label="Start Date")
        start_hour = gr.Dropdown(label="Start Hour", choices=[str(i) for i in range(24)])
        start_minute = gr.Dropdown(label="Start Minute", choices=[str(i) for i in range(60)])
        start_second = gr.Dropdown(label="Start Second", choices=[str(i) for i in range(60)])
    
    with gr.Row():
        end_date = Calendar(type="datetime", label="End Date")
        end_hour = gr.Dropdown(label="End Hour", choices=[str(i) for i in range(24)])
        end_minute = gr.Dropdown(label="End Minute", choices=[str(i) for i in range(60)])
        end_second = gr.Dropdown(label="End Second", choices=[str(i) for i in range(60)])
    
    sensor_name_1 = gr.Dropdown(label="Sensor Name", choices=sensor_names)
    
    summary_output = gr.Textbox(label="Video Summary", lines=10)
    
    def process_summary_and_frames(*args):
        summary = load_video_summary(*args)
        return summary
    
    submit_btn = gr.Button("Fetch Summary")
    submit_btn.click(fn=process_summary_and_frames, inputs=[start_date, start_hour, start_minute, start_second, end_date, end_hour, end_minute, end_second, sensor_name_1], outputs=summary_output)
    
    gr.Markdown("# Daily Digest")
    gr.Markdown("Pick a date to fetch the daily digest for the camera.")
    
    digest_date = Calendar(type="date", label="Digest Date")
    sensor_name_2 = gr.Dropdown(label="Sensor Name", choices=sensor_names)

    digest_output = gr.Textbox(label="Daily Digest", lines=10)
    
    digest_btn = gr.Button("Fetch Daily Digest")
    digest_btn.click(fn=load_daily_digest, inputs=[digest_date, sensor_name_2], outputs=digest_output)

demo.launch()