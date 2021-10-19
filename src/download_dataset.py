import argparse
import glob
import json
import sys
import zipfile
import pandas as df
import cv2
import pandas as pd
import requests
import os
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data', help='The path where to save downloaded dataset')

URL = "https://bitbucket.org/merayxu/multiview-object-tracking-dataset/get/59d1683b271b.zip"


def split_annotations(annotations, test_size=0.2, random_state=0):
    # split dataset
    training_annotations, testing_annotations = {}, {}
    # Build target
    for key, data in annotations.items():
        training_annotations[key] = []
        train_frames, test_frames = train_test_split(data, stratify=[x[-1] for x in data], test_size=test_size,
                                                     shuffle=True, random_state=random_state)
        training_annotations[key] = train_frames
        testing_annotations[key] = test_frames
    return training_annotations, testing_annotations


def batch_iou(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def build_target(annotations, iou_tolerance=0.1):
    def compare_bbox(info, prev_info, iou_tolerance=iou_tolerance):
        if info is None or prev_info is None:
            return False
        if info.shape != prev_info.shape:
            return False
        if not np.all(info.track_id.values == prev_info.track_id.values):
            return False
        a = info[['xmin', 'ymin', 'xmax', 'ymax']].values
        b = prev_info[['xmin', 'ymin', 'xmax', 'ymax']].values
        iou = 1 - batch_iou(a, b)
        if np.all(iou <= iou_tolerance):
            return True
        return False

    metadata = {}
    count0 = 0
    count1 = 0
    # Build target
    for idx, (key, data) in enumerate(annotations.items()):
        metadata[key] = []
        frames_nums = sorted(list(data.keys()), key=lambda x: int(x))
        base_frame_info = None
        base_frame_num = frames_nums[0]
        for num in frames_nums:
            frame_info = data[num]
            columns = ['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'lost', 'occluded', 'generated', 'label']
            frame_info = pd.DataFrame(frame_info, columns=columns)
            frame_info = frame_info.loc[frame_info.label == 'PERSON']
            frame_info = frame_info.loc[frame_info.lost == 0]
            frame_info = frame_info.sort_values(by=['track_id'])

            if compare_bbox(frame_info, base_frame_info, iou_tolerance=iou_tolerance):
                target = 1
                metadata[key].append((base_frame_num, num, target))
                count1 += 1
            else:
                target = 0
                metadata[key].append((base_frame_num, num, target))
                base_frame_info = frame_info
                base_frame_num = num
                count0 += 1
            sys.stdout.write(f"Class1 : {count1} | Class0 : {count0}\r")
            sys.stdout.flush()

        # print()
    sys.stdout.write(f"Class1 : {count1} | Class0 : {count0}\r")
    sys.stdout.flush()
    return metadata


if __name__ == '__main__':
    args = parser.parse_args()
    path = os.path.join(args.data_path, "../data/data.zip")
    frames_dir = os.path.join(args.data_path, "../data/frames")
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    download = True
    un_zip = True
    extract_frames = True
    data_split = True
    WIDHT, HEIGHT = 112, 112

    # Download the video dataset
    if download:
        r = requests.get(URL, stream=True)
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            n_chunks = (total_length / 1024) + 1
            for chunk in tqdm(r.iter_content(chunk_size=1024), total=n_chunks):
                if chunk:
                    f.write(chunk)
                    f.flush()
    # Extract video file
    if un_zip:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(args.data_path)

    # Video to frames
    if extract_frames:
        path = os.path.join(args.data_path, "../data/merayxu-multiview-object-tracking-dataset-59d1683b271b/CAMPUS")
        dirs = os.listdir(path)
        annotations_data = {}
        print(dirs)
        for dir_name in dirs:
            videos = sorted(glob.glob(os.path.join(path, dir_name, '*.mp4')))
            annotations = sorted(glob.glob(os.path.join(path, dir_name, '*.txt')))
            # os.makedirs(os.path.join(frames_dir, dir_name), exist_ok=True)
            print(f"DIR : {dir_name}")
            for source, annotation in zip(videos, annotations):
                header = ['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame_number', 'lost', 'occluded', 'generated',
                          'label']
                ann = df.read_csv(annotation, delimiter=" ", header=None, names=header)
                cap = cv2.VideoCapture(source)
                basename = os.path.basename(source)
                basename = os.path.splitext(basename)[0]
                i = 0
                last_frame = None
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # resize image
                    last_frame = frame
                    frame = cv2.resize(frame, (WIDHT, HEIGHT), interpolation=cv2.INTER_AREA)
                    frame_path = os.path.join(frames_dir, f"{dir_name}_{basename}_{i}.png")
                    cv2.imwrite(frame_path, frame)
                    print(f"{i}-->{basename}", end="\r")
                    i += 1
                default_width, default_height, _ = last_frame.shape
                ann.ymin = (ann.ymin / default_width) * WIDHT
                ann.ymax = (ann.ymax / default_width) * WIDHT
                ann.xmin = (ann.xmin / default_height) * HEIGHT
                ann.xmax = (ann.xmax / default_height) * HEIGHT
                header = ['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'lost', 'occluded', 'generated', 'label']
                ann = ann.groupby('frame_number')[header].apply(lambda g: g.values.tolist()).to_dict()
                print(f"({dir_name},{basename}): ", i - 1, len(ann))
                annotations_data[f"{dir_name}_{basename}"] = ann
                print()
        # save the sequences data
        json.dump(annotations_data, open(os.path.join(os.path.join(frames_dir, 'annotations.json')), 'w'))

    if data_split:
        anno_path = os.path.join(os.path.join(frames_dir, 'annotations.json'))
        annotations = json.load(open(anno_path, 'r'))
        print("Build Targets ... ")
        annotations = build_target(annotations)
        print("Split Training/Testing dataset...")
        training_annotations, testing_annotations = split_annotations(annotations, test_size=0.2)

        json.dump(training_annotations, open(os.path.join(os.path.join(frames_dir, 'training_annotations.json')), 'w'))
        json.dump(testing_annotations, open(os.path.join(os.path.join(frames_dir, 'testing_annotations.json')), 'w'))