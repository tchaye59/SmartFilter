# run this program on the Mac to display image streams from multiple RPis
import argparse

import cv2
import imagezmq
import numpy as np
from utils import faster_rcnn_predict, draw_bboxes, EpisodesPool, OnlinePPOAdapter, ssd_mobilenet_predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receiver_port', type=int, default=5555,
                        help='The port used by camera devices to send frames')

    parser.add_argument('--pool_size', type=int, default=1000, help='')

    parser.add_argument('--episode_size', type=int, default=10, help='')

    parser.add_argument('--sender_port', type=int, default=6666,
                        help='The port used by the client by the client to access the results')
    parser.add_argument('--visualize', type=bool, default=False, help='Visualise or not')
    parser.add_argument('--adapt', type=bool, default=False, help='Run the online adaptation for each camera unit')
    parser.add_argument('--stream_to_client', type=bool, default=False, help='')
    parser.add_argument('--jpeg_quality', type=int, default=95,
                        help='0 to 100, higher is better quality, 95 is cv2 default')
    parser.add_argument('--log_path', type=str, default="logs")
    parser.add_argument('--model_path', type=str, default="models/")

    args = parser.parse_args()
    return args


def put_frame(frame, bboxes, adapters):
    if key not in adapters:
        adapter = OnlinePPOAdapter(EpisodesPool(window_size=args.episode_size, pool_size=args.pool_size), key)
        adapters[key] = adapter
        adapter.start()

    adapter = adapters[key]
    adapter.pool.put(frame, bboxes)


# Try detectron
bboxes = faster_rcnn_predict(np.zeros((112, 112, 3)), target_class=0)

if __name__ == "__main__":
    args = parse_args()
    image_hub = imagezmq.ImageHub(open_port=f'tcp://*:{args.receiver_port}', )
    # Create an image sender in PUB/SUB (non-blocking) mode
    sender = imagezmq.ImageSender(connect_to=f'tcp://*:{args.sender_port}', REQ_REP=False)
    print("Connected..")

    adapters = {}

    while True:  # show streamed images until Ctrl-C
        key, jpg_buffer = image_hub.recv_jpg()
        # Decode the image
        image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)

        bboxes = faster_rcnn_predict(image, target_class=0)
        # bboxes = ssd_mobilenet_predict(image, target_class=1)
        result_image = draw_bboxes(image, bboxes)

        if args.adapt:
            put_frame(image, bboxes, adapters)

        if args.visualize:
            cv2.imshow(key, result_image)  # 1 window for each RPi

        image_hub.send_reply(b'OK')

        if args.stream_to_client:
            # Encode the image for lower bandwidth usage with the client
            ret_code, jpg_buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            # Send an image to the queue(The client will read it)
            sender.send_jpg(key, jpg_buffer)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
