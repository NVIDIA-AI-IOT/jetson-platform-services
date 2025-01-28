# Video Summarization Microservice

## Overview

The Video Summarization Microservice is an AI NVR service component within the Jetson Platform Services. It is designed to analyze both live video streams and snippets, generating summaries of human activity. By leveraging state-of-the-art large models, including VILA1.5 and GPT-4, the microservice provides valuable insights into video content, particularly useful in surveillance, security, and behavioral analysis applications.

For comprehensive documentation about the Video Summarization Microservice, view the content below to get started.

## Key Components

1. **`api_server.py`**: Acts as the interface between clients and the microservice, handling API requests, managing the Redis database for video summaries, and coordinating data flow. Built with Flask for scalability and handling multiple requests.
2. **`main.py`**: Orchestrates the video processing pipeline, integrating video streaming, AI model inference, and API server interaction. It uses computer vision to extract frames and generate summaries of human activity.
3. **`config/main_config.json`**: Central configuration file for customizing system parameters, including API ports, Redis settings, and AI model options.
4. **Useful Tools**: 
   - **`app.py`**: Fetches and downloads video summaries and snippets by interacting with the API server. It efficiently retrieves and downloads relevant video segments for user review and analysis.
   - **`docker_start.py`**: Designed for stream-mode processing on station, directly adding all available streams, then sending real-time summary requests, and restarting the service at regular intervals to ensure continuous stream tracking.



## Video Summarization Microservice

The Video Summarization Microservice is designed for efficient video file summarization on Jetson devices.

### 1. Build the Docker Image

Use the following command to build the Docker image:

```bash
docker build -t video_summarization_microservice .
```

### 2. Start the Microservice

Make sure the Redis service is running:

```bash
sudo systemctl start jetson-redis
```

Modify the config/main_config.json file as needed. Make sure that "openai_api_key" is properly set.

Launch the microservice using the command:

```bash
docker run -itd --runtime nvidia --network host -v ./config/main_config.json:/configs/main_config.json -v /data/vsm-videos/:/data/videos/ -e CONFIG_PATH="/configs/main_config.json" video_summarization_microservice
```

Here we are mounting the config file within the container. We then also mount a directory from the host (/data/vsm-videos which can be changed) into the container - this directory is where your video files should be stored/copied to. We then define the environment variable which specifies where your config file is within the container.  


Once launched, view the logs::

```bash
docker logs -f <docker id>
```

Once you see something similar to below, the container is up and ready to accept incoming requests:
```bash
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:19000 (Press CTRL+C to quit)
```

### 3. Upload Video Files

Upload video files using the following `curl` command:

```bash
curl -X POST "http://localhost:19000/files?filepath=demo.mp4"
```

Filepath is based on where the file is within the container. 

### 4. Request Summarization

Request a summarization of the video using:

```bash
curl -X POST "http://localhost:19000/summarize" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{"stream": false, "id": "93725c66-f8bf-431a-b16f-9a21a5e7bbc3", "model": "vila-1.5", "chunk_duration": 20}'
```

id should be the id returned back from step 3

### 5. Fetch Summarization Results

Retrieve the summarization results with:

```bash
curl -X GET "http://localhost:19000/summarize?stream=false&id=01fc1241-27a2-4f91-a771-bcb65d0846ba"
```

id should be the id returned back from step 3


## Configuration

The `main_config.json` file offers extensive customization options for the microservice. Key configurations include:

- **`api_server_port`**: Defines the port on which the API server listens for requests.
- **`redis_host`**: Specifies the IP address of the Redis database used for data storage.
- **`redis_port`**: Sets the port number for the Redis database.
- **`redis_stream`**: Names the stream within the Redis database used for storing data.
- **`log_level`**: Determines the level of log output.
- **`jetson_ip`**: The IP address of the Jetson device (if applicable) for edge computing deployments.
- **`video_port`**: Port utilized for video streaming services.
- **`max_images`**: Specifies the maximum number of images processed per request, helping manage resource utilization.
- **`ingress_port`**: Port designated for data ingestion from external sources.
- **`segment_time`**: The time interval for processing video segments, balancing real-time responsiveness with processing efficiency.



# **Video Summarization Microservice API Guide**

The video summaries and daily reports microservice are accessible via a RESTful API. Below are the details for accessing and utilizing the Video Summary API endpoints provided by the FastAPI server.

## **API Base URL**

The base URL for the API is:

`http://<HOST-IP>:<PORT>/`

Replace `<HOST-IP>` with the server's IP address and `<PORT>` with the port number on which the server is running (default is 19000).

## **Endpoints**

### **1. Health Check**

Provides the readiness and liveness status of the service.

#### **Readiness Check**

**Endpoint:**

`HTTP GET /health/ready`

**Description:**

Checks if the service is ready to handle requests.

**Response:**

- A successful response will have a 200 status code and return a JSON object indicating the service's readiness status.

```json
{
  "status": "ready"
}
```

##### Example:

```
curl -X GET "http://localhost:19000/health/ready" -H "accept: application/json"
```

#### **Liveness Check**

**Endpoint:**

`HTTP GET /health/live`

**Description:**

Checks if the service is alive.

**Response:**

- A successful response will have a 200 status code and return a JSON object indicating the service's liveness status.

```json
{
  "status": "live"
}
```

##### Example:

```
curl -X GET "http://localhost:19000/health/live" -H "accept: application/json"
```

---

### 2. **Video File Management**

Manage video files by uploading, listing, retrieving, and deleting them.

#### **Upload a Video File**

**Endpoint:**

`HTTP POST /file`

**Description:**

Upload a video file to the system and store its metadata.

**Request Body:**

- `filepath`
  - **Description:** The file path of the video to be uploaded.
  - **Type:** string
  - **Example:** "/videos/demo.mp4"

**Response:**

- A successful response will have a 201 status code and return the unique `file_id`.

```json
{
  "file_id": "string"
}
```

##### Example:

```bash
curl -X POST "http://localhost:19000/files?filepath=/data/via_jetson/demo.mp4"
```

#### **List All Video Files**

**Endpoint:**

`HTTP GET /files`

**Description:**

Retrieve a list of all uploaded video files.

**Response:**

- A successful response will have a 200 status code and return a JSON object with file IDs and their corresponding file paths.

```json
{
  "file_id1": {
    "filepath": "/videos/demo1.mp4"
  },
  "file_id2": {
    "filepath": "/videos/demo2.mp4"
  }
}
```

##### Example:

```bash
curl -X GET http://localhost:19000/files
```

#### **Retrieve a Video File**

**Endpoint:**

`HTTP GET /file/{file_id}`

**Description:**

Retrieve metadata for a specific video file using its `file_id`.

**Path Parameter:**

- `file_id`
  - **Description:** Unique identifier of the video file to be retrieved.
  - **Type:** string
  - **Example:** "93725c66-f8bf-431a-b16f-9a21a5e7bbc3"

**Response:**

- A successful response will have a 200 status code and return the `file_id` and `filepath`.

```json
{
  "file_id": "string",
  "filepath": "/videos/demo.mp4"
}
```

##### Example:

```bash
curl -X GET http://localhost:19000/files/93725c66-f8bf-431a-b16f-9a21a5e7bbc3
```

#### **Delete a Video File**

**Endpoint:**

`HTTP DELETE /file/{file_id}`

**Description:**

Delete a specific video file using its `file_id`.

**Path Parameter:**

- `file_id`
  - **Description:** Unique identifier of the video file to be deleted.
  - **Type:** string
  - **Example:** "93725c66-f8bf-431a-b16f-9a21a5e7bbc3"

**Response:**

- A successful response will have a 200 status code and a message indicating the file was deleted.

```json
{
  "status": "success",
  "message": "File with ID 93725c66-f8bf-431a-b16f-9a21a5e7bbc3 deleted."
}
```

- If the `file_id` is not found, the response will have a status of 404 with an error message.

```json
{
  "status": "error",
  "message": "File with ID 93725c66-f8bf-431a-b16f-9a21a5e7bbc3 not found."
}
```

##### Example:

```bash
curl -X DELETE http://localhost:19000/files/01fc1241-27a2-4f91-a771-bcb65d0846ba -H "accept: application/json"
```

---

### **3. Model Management**

Retrieve the list of available models.

#### **List Models**

**Endpoint:**

`HTTP GET /models`

**Description:**

Fetches a list of available models that can be used for processing live streams or summarization.

**Response:**

- A successful response will have a 200 status code and return a JSON array of model objects.

```json
[
  {
    "model_id": "string",
    "model_name": "string",
    "description": "string"
  }
]
```

##### Example:

```
curl -X GET "http://localhost:19000/models" -H "accept: application/json"
```

---

### **4. Video Summarization**

Generate summaries for specific segments of a live stream.

#### **Create a Video Summary**

**Endpoint:**

```
HTTP POST /summarize
```

**Request Body:**

```json
{
  "id": "string",
  "prompt": "string",
  "stream_id": "string",
  "model": "string",
  "stream": "boolean",
  "chunk_duration": "integer",
  "from_timestamp": "string",
  "to_timestamp": "string"
}
```

**Mandatory Fields:**

- model
  - **Description:** The model to use for summarization.
  - **Type:** string
  - **Example:** "vila-1.5"

**Optional Fields:**

- `id`
  - **Description:** Identifier for the video file.
  - **Type:** string
  - **Example:** "123e4567-e89b-12d3-a456-426614174000"
- `prompt`
  - **Description:** Optional text prompt to guide the summarization.
  - **Type:** string
  - **Example:** "Summarize the key events from the stream."
- `stream_id`
  - **Description:** Identifier for the video stream.
  - **Type:** string
  - **Example:** "1d2a0309-ec34-4713-866f-44ee2b830443"
- `stream`
  - **Description:** Indicates if the summarization is for a live stream.
  - **Type:** boolean
  - **Default:** false
- `chunk_duration`
  - **Description:** Duration of each chunk of a file to be summarized, in seconds.
  - **Type:** integer
  - **Default:** 20
- `from_timestamp`
  - **Description:** Start time of the range for which video snippets are to be summarized.
  - **Type:** UTC / GMT timestamp string
  - **Example:** "2024-08-16T21:00:00.270Z"
- `to_timestamp`
  - **Description:** End time of the range for which video snippets are to be summarized.
  - **Type:** UTC / GMT timestamp string
  - **Example:** "2024-08-16T23:25:44.384Z"

**Response:**

- A successful response will have a 200 status code. The response body will contain the summary of the video snippets within the provided time range.

```json
{
  "summary_id": "string",
  "stream_id": "string",
  "model": "string",
  "from_timestamp": "string",
  "to_timestamp": "string",
  "snippets": [
    {
      "start_time": "string",
      "end_time": "string",
      "summary": "Description of the video content"
    }
  ]
}
```

##### Example:

Summarization Request for a Live-stream.

```
curl -X POST "http://localhost:19000/summarize" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{"stream": true, "stream_id": "1d2a0309-ec34-4713-866f-44ee2b830443", "model": "vila-1.5", "chunk_duration": 20, "from_timestamp": "2024-08-23T16:00:00.000Z", "to_timestamp": "2024-08-23T16:22:00.000Z"}'

```

Summarization Request for a Video File.

```
curl -X POST "http://localhost:19000/summarize" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{"stream": false, "id": "93725c66-f8bf-431a-b16f-9a21a5e7bbc3", "model": "vila-1.5", "chunk_duration": 20}'
```

#### **Fetch a Video Summary**

Retrieve summaries for specific segments of a live stream or a particular summary by ID.

**Endpoint:**

```
HTTP GET /summarize
```

**Request Parameters:**

- `stream` (optional)
  - **Description:** Indicates if the summarization is for a live stream.
  - **Type:** boolean
  - **Default:** false
  - **Example:** true
- `id` (optional)
  - **Description:** Unique identifier for the specific summarization request.
  - **Type:** string
  - **Example:** "123e4567-e89b-12d3-a456-426614174000"
- `stream_id` (optional)
  - **Description:** Identifier for the video stream.
  - **Type:** string
  - **Example:** "1d2a0309-ec34-4713-866f-44ee2b830443"
- `from_timestamp` (optional)
  - **Description:** Start time of the range for which video snippets are to be summarized.
  - **Type:** UTC / GMT timestamp string
  - **Example:** "2024-08-16T21:00:00.000Z"
- `to_timestamp` (optional)
  - **Description:** End time of the range for which video snippets are to be summarized.
  - **Type:** UTC / GMT timestamp string
  - **Example:** "2024-08-16T23:00:44.384Z"

**Response:**

- A successful response will have a 200 status code and return a summary of the video snippets detected within the provided time range.

```json
{
  "summary_id": "string",
  "stream_id": "string",
  "from_timestamp": "string",
  "to_timestamp": "string",
  "snippets": [
    {
      "start_time": "string",
      "end_time": "string",
      "summary": "Description of the video content"
    }
  ]
}
```

##### Example:

Summarization Fetch Request for a Live-stream.

```
curl -X GET "http://localhost:19000/summarize?stream=true&stream_id=1d2a0309-ec34-4713-866f-44ee2b830443&from_timestamp=2024-08-16T04:32:44.384Z&to_timestamp=2024-08-16T17:31:04.384Z" -H "accept: application/json"
```

Summarization Fetch Request for a Video File.

```
curl -X GET "http://localhost:19000/summarize?stream=false&id=01fc1241-27a2-4f91-a771-bcb65d0846ba"
```

#### **Retrieve a Stream Daily Digest**

**Endpoint:**

`HTTP GET /summarize/digest`

**Request Parameters:**

- `stream_id`
  - **Description:** Identifier for the video stream.
  - **Type:** string
  - **Example:** "1d2a0309-ec34-4713-866f-44ee2b830443"

- `date`
  - **Description:** Date for which the digest summary is to be retrieved.
  - **Type:** date string in `YYYY-MM-DD` format
  - **Example:** "2024-08-13"

**Response:**

- A successful response will have a 200 status code and return a digest summary of the video content for the specified date.

```json
{
  "digest_id": "string",
  "stream_id": "string",
  "date": "string",
  "summary": "Digest summary of the video content"
}
```

##### Example:

```
curl -X GET "http://localhost:19000/summarize/digest?stream_id=1d2a0309-ec34-4713-866f-44ee2b830443&date=2024-08-13" -H "accept: application/json"
```
