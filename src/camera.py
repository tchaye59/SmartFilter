import argparse
import os
import socket

import cv2
import imagezmq
import numpy as np

from src.model import build_actor, to_TFLite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--video', type=str, default="", help='Use a video file instead of the camera device')
    parser.add_argument('--camera_id', type=int, default=0, help='Which camera to use')
    parser.add_argument('--visualize', type=int, default=False, help='Visualise or not')
    parser.add_argument('--stream', type=int, default=True, help='Stream video data to server or not')
    parser.add_argument('--filter', type=bool, default=False, help='Run filtering or not')
    parser.add_argument('--verbose', type=bool, default=True, help='Print logs')
    parser.add_argument('--model_path', type=str, default="models/actor_actor.h5", help='')
    parser.add_argument('--jpeg_quality', type=int, default=95, help='0 to 100, higher is better quality, 95 is cv2 '
                                                                     'default')
    parser.add_argument('--win_size', type=int, default=100, help='The frames window size')

    args = parser.parse_args()
    return args


def smart_filter(frame, model_path,win_size=100):
    frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    if not hasattr(smart_filter, 'data'):
        actor = build_actor()
        actor.load_weights(model_path)
        actor = to_TFLite(actor)
        win_counter = 0
        smart_filter.data = actor, frame, np.array([0, ]),win_counter
        return 0
    actor, base_frame, prev_action,win_counter= smart_filter.data

    if win_counter >= win_size:
        # Reset the window
        base_frame = frame
        action = np.array([0, ])
        win_counter = 0
    else:
        # Predict
        X = ((frame - base_frame)[np.newaxis], prev_action.reshape((1, 1)))
        prob = actor(X)
        action = np.argmax(prob, axis=-1)

    if action[0] == 0:
        base_frame = frame
    win_counter += 1
    smart_filter.data = actor, base_frame, action,win_counter
    return action[0]


def run():
    args = parse_args()

    input_video = args.video if os.path.exists(args.video) else None
    camera_id = args.camera_id
    visualize = args.visualize
    stream = args.stream
    ip = args.ip
    port = args.port
    win_size = args.win_size
    source = input_video if input_video else camera_id
    sender = imagezmq.ImageSender(connect_to=f'tcp://{ip}:{port}')
    message = socket.gethostname()  # send hostname with each image
    cap = cv2.VideoCapture(source)
    model_path = args.model_path
    dropped_frames = 0
    sent_frames = 0

    while cap.isOpened():

        ret, frame = cap.read()
        # Encode the image for lower bandwidth usage
        ret_code, jpg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

        if not ret:
            continue
        if not stream and visualize:
            cv2.imshow(message, frame)

        if stream:
            if args.filter:
                action = smart_filter(frame, model_path,win_size=win_size)
                if action == 0:
                    if visualize:
                        cv2.imshow(message, frame)
                    sender.send_jpg(message, jpg_buffer)
                    sent_frames += 1
                else:
                    dropped_frames += 1
            else:
                sender.send_jpg(message, jpg_buffer)
                sent_frames += 1
        else:
            dropped_frames += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if args.verbose:
            print(f">>>>>Dropped Frames: {dropped_frames} | Sent Frames: {sent_frames}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(">>>>>Starting the camera...")
    run()
