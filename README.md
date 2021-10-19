# SmartFilter


## How to run SmartFilter
1. Clone the repo and make sure you have installed the require dependancies from the `requirements.txt` file. For GPU users make sure you install the GPU version of the following libraries: `tensorflow,torch and detectron2`
2. Run following commands to download and prepare the video dataset for supervised traning
```bash
cd  ./SmartFilter
python -m   src.download_dataset --data_path ./data
```

`./data` is the path to where the downloaded data will be stored 


##  Supervised Training
```
 python -m  src.supervised_training --data_path ./data
```

##  Online Adaptation

1. Start the server:
```
    python -m src.server --adapt True 
```
2. Stream from the camera to server
```
    python -m src.camera --ip SERVER_IP
```
3. Stream a pre-recoded video to the server
```
    python -m src.camera --ip SERVER_IP --video ./video.mp4
```


##  Live Filtering
1. Server:
```
    python -m src.server --stream_to_client True 
```
2. Camera:
```
    python -m src.camera --filter True 
```
3. Client:
```
    python -m src.client --ip SERVER_IP
```