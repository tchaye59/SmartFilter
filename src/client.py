# run this program on the Mac to display image streams from multiple RPis
import argparse
import cv2
import imagezmq
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6666)
    parser.add_argument('--ip', type=str, default='localhost')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    image_hub = imagezmq.ImageHub(open_port=f'tcp://{args.ip}:{args.port}', REQ_REP=False)
    print("Connected..")

    while True:  # show streamed images until Ctrl-C
        message, jpg_buffer = image_hub.recv_jpg()

        # Decode the image
        image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)

        cv2.imshow(message, image)  # 1 window for each camera
        print(message)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
