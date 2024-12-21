#!/bin/bash

# Function to run api_server.py and save output to log
run_main() {
  python3 ./src/api_server.py > api_server.log 2>&1 &
}

# Function to send POST requests to the live stream API
send_live_stream_requests() {
  curl -X POST "http://localhost:19000/live-stream" -H "accept: application/json" -H "Content-Type: application/json" -d '{ "liveStreamUrl": "1d2a0309-ec34-4713-866f-44ee2b830443", "description": "Test live stream"}'
  curl -X POST "http://localhost:19000/live-stream" -H "accept: application/json" -H "Content-Type: application/json" -d '{ "liveStreamUrl": "1d2fbca1-ef12-43af-a3ec-e26bdf75bc1e", "description": "Test live stream"}'
  curl -X POST "http://localhost:19000/live-stream" -H "accept: application/json" -H "Content-Type: application/json" -d '{ "liveStreamUrl": "2cb6a3b4-3f88-4907-b3b3-0b5c389416bb", "description": "Test live stream"}'
  curl -X POST "http://localhost:19000/live-stream" -H "accept: application/json" -H "Content-Type: application/json" -d '{ "liveStreamUrl": "53cb71f9-7ccb-4f3b-847f-dc93aff8d1dc", "description": "Test live stream"}'
  curl -X POST "http://localhost:19000/live-stream" -H "accept: application/json" -H "Content-Type: application/json" -d '{ "liveStreamUrl": "a2b24809-aa28-493b-b08e-af28d3c79530", "description": "Test live stream"}'
  curl -X POST "http://localhost:19000/live-stream" -H "accept: application/json" -H "Content-Type: application/json" -d '{ "liveStreamUrl": "d83797a0-239c-47b5-8923-dea1202684fd", "description": "Test live stream"}'
  curl -X POST "http://localhost:19000/live-stream" -H "accept: application/json" -H "Content-Type: application/json" -d '{ "liveStreamUrl": "e5b12f7d-1c55-4812-89a7-ffc3a10e4e40", "description": "Test live stream"}'
}

# Function to send summarize requests for each stream_id with dynamic timestamps
send_summarize_requests() {
  from_timestamp=$(date -u -d '20 minutes ago' +"%Y-%m-%dT%H:%M:%S.%3NZ")
  to_timestamp=$(date -u -d '15 minutes' +"%Y-%m-%dT%H:%M:%S.%3NZ")
  
  stream_ids=(
    "1d2a0309-ec34-4713-866f-44ee2b830443"
    "1d2fbca1-ef12-43af-a3ec-e26bdf75bc1e"
    "2cb6a3b4-3f88-4907-b3b3-0b5c389416bb"
    "53cb71f9-7ccb-4f3b-847f-dc93aff8d1dc"
    "a2b24809-aa28-493b-b08e-af28d3c79530"
    "d83797a0-239c-47b5-8923-dea1202684fd"
    "e5b12f7d-1c55-4812-89a7-ffc3a10e4e40"
  )

  for stream_id in "${stream_ids[@]}"; do
    curl -X POST "http://localhost:19000/summarize" -H "accept: application/json" -H "Content-Type: application/json" -d '{
      "stream_id": "'"${stream_id}"'", 
      "model": "vila-1.5", 
      "from_timestamp": "'"${from_timestamp}"'", 
      "to_timestamp": "'"${to_timestamp}"'"
    }'
  done
}

# Function to send clustering requests for each stream_id with the same timestamps
send_clustering_requests() {
  from_timestamp=$(date -u -d '60 minutes ago' +"%Y-%m-%dT%H:%M:%S.%3NZ")
  to_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
  
  stream_ids=(
    "1d2a0309-ec34-4713-866f-44ee2b830443"
    "1d2fbca1-ef12-43af-a3ec-e26bdf75bc1e"
    "2cb6a3b4-3f88-4907-b3b3-0b5c389416bb"
    "53cb71f9-7ccb-4f3b-847f-dc93aff8d1dc"
    "a2b24809-aa28-493b-b08e-af28d3c79530"
    "d83797a0-239c-47b5-8923-dea1202684fd"
    "e5b12f7d-1c55-4812-89a7-ffc3a10e4e40"
  )

  for stream_id in "${stream_ids[@]}"; do
    curl -X POST "http://localhost:19000/clustering" -H "accept: application/json" -H "Content-Type: application/json" -d '{
      "stream_id": "'"${stream_id}"'", 
      "from_timestamp": "'"${from_timestamp}"'", 
      "to_timestamp": "'"${to_timestamp}"'"
    }'
  done
}

# send_clustering_requests

while true; do
  run_main
  sleep 100

  # Run the requests to the live stream API
  send_live_stream_requests
  
  # Run the summarize requests for each stream_id
  send_summarize_requests
  
  # Sleep for 1800 seconds (30 minutes)
  sleep 1800
  
  # Run the clustering requests for each stream_id
  # send_clustering_requests
  # wait

  # Terminate the background processes started by run_main
  pkill -P $$
done
