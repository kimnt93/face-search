# https://githubhot.com/repo/zakerifahimeh/FaceLib
import json
import time
import logging
import cv2
from lib.indexer import MilvusIndexer
import multiprocessing
import threading
import cachetools.func
import datetime
import copy
from lib.models import *
from lib import config
import os

from lib.utils import save_frame, load_image, create_full_default_left, get_now_approx

LOGGER = logging.getLogger()

desired_width = 640
desired_height = 480
video_capture = cv2.VideoCapture(config["video_capture"])
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
# set W and H
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# crop_width = crop_height = min(frame_width, frame_height)

crop_width = frame_width // 4 * 3
crop_height = frame_height // 4 * 3

start_x = (frame_width - crop_width) // 2
start_y = (frame_height - crop_height) // 2

# Define the region of interest (ROI) rectangle
roi = (start_x, start_y, crop_width, crop_height)


scale_ratio = config['scale_ratio']
scale_size = 1 / scale_ratio
MAX_IMAGE_ON_LEFT = 6
SHOW_RELATED_EVERY_SECONDS = 0.5

left_height = crop_height
left_width = crop_width
left_image_shape = (left_width, left_height)
left_image_single_shape = (int(left_height), int(left_width / 2))
REPLAY_FPS = 26
recognize_fps = config['recognize_fps']
face_model = globals()[config['face_model']](scale_ratio=scale_ratio)  # FaceRecognitionWithMask(scale_ratio=scale_ratio)

# shared 
## # 
emb_queue = multiprocessing.Queue()
show_live_queue = multiprocessing.Queue()
notify_queue = multiprocessing.Queue()
show_notify_queue = multiprocessing.Queue()

interval = 1 / REPLAY_FPS


def search_and_notify():
    ndim = 512 if face_model.__class__.__name__ == "FaceRecognitionWithMask" else 128
    if config['threshold'] != "auto":
        threshold = config['threshold']
    else:
        if config['face_model'] == "FaceRecognitionSimple":
            threshold = 0.9
        else:
            threshold = 0.75

    idx = MilvusIndexer(collection_name=config['face_model'], ndim=ndim, remove=config['remove_old'], threshold=threshold)
    prev_time = get_now_approx()
    now = get_now_approx()
    search_result = dict()
    while True:
        # now = get_now_approx()
        data = notify_queue.get()

        # get embeddings
        frame_id = data['frame_id']
        # get embedding and normalize
        embeddings = data['embeddings']

        # 2022-04-24 15:45:14,902 - search_and_notify - INFO -
        # Search result
        # {
        #   1650789914691: {'frame': 1650789914691, 'score': 0.9559153914451599},
        #   1650789884377: {'frame': 1650789884377, 'score': 0.8205408751964569}
        # }
        emb_rs = idx.search(now, embeddings)
        if emb_rs:
            search_result.update(emb_rs)

        # LOGGER.info(f"Search result {rs}")
        idx.insert(frame_id, embeddings)

        now = get_now_approx()
        if search_result and (now - prev_time).total_seconds() > SHOW_RELATED_EVERY_SECONDS:
            prev_time = get_now_approx()
            LOGGER.info(f"List related faces: {search_result}")
            show_notify_queue.put(search_result)
            search_result = dict()


def show_related_im():
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        LOGGER.info("============ SHOW RELATED IMAGES ============")
        data = show_notify_queue.get()
        images = list()
        data = sorted(data.items(), key=lambda x: x[1]['score'], reverse=True)

        for frame_id, frame_data in data[:MAX_IMAGE_ON_LEFT]:  # top
            LOGGER.info(f"Load frame and add {frame_id} : {frame_data}")
            score = frame_data['score']
            score = round(score * 100, 2)
            im = load_image(frame_id)
            now = datetime.datetime.fromtimestamp(frame_id / 1000).strftime("%H:%M:%S")
            cv2.putText(im, str(now), (55, 110), font, 0.44, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(im, str(score) + "%", (5, 20), font, 0.44, (255, 0, 0), 1, cv2.LINE_AA)

            images.append(im)

        im_related = create_left_im(images)
        LOGGER.info(f"Send related faces image: {im_related.shape}")
        show_live_queue.put({
            "origin": False,
            "frame": im_related
        })


def create_left_im(images):
    def create_im(i1, i2):
        # i1 = cv2.resize(i1, left_image_single_shape)
        # i2 = cv2.resize(i2, left_image_single_shape)
        ir = cv2.hconcat([i1, i2])
        return ir

    total_im = len(images)
    if total_im <= 0:
        im_final = create_full_default_left()
    else:
        # generate cho du hinh ne
        for i in range(total_im, MAX_IMAGE_ON_LEFT):
            images.append(images[-1])

        imf_01 = create_im(images[0], images[1])
        imf_02 = create_im(images[2], images[3])
        imf_03 = create_im(images[4], images[5])
        im_final = cv2.vconcat([imf_01, imf_02])
        im_final = cv2.vconcat([im_final, imf_03])

    return im_final


def show_live_frames():
    cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    im_left = create_full_default_left()
    data = show_live_queue.get()
    frame = data['frame']
    im_left = cv2.resize(im_left, (im_left.shape[1], frame.shape[0]))

    while True:
        data = show_live_queue.get()
        if not data['origin']:
            im_left = data['frame']
            im_left = cv2.resize(im_left, (im_left.shape[1], frame.shape[0]))
        else:
            frame = data['frame']

        im_final = cv2.hconcat([frame, im_left])
        cv2.imshow('Video', im_final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            os._exit(1)
            break

        # time.sleep(interval)

    cv2.destroyAllWindows()


def start_stream():
    LOGGER.info("Start camera...")
    i = 0
    #
    si = int(REPLAY_FPS / recognize_fps)
    while video_capture.isOpened():
        _, frame = video_capture.read()

        if frame is None:
            continue
        roi_frame = frame[start_y:start_y+crop_height, start_x:start_x+crop_width]
        # roi_frame = frame
        if i % si == 0:
            roi_frame, small_frame, embeddings = face_model.get_faces(roi_frame)
            if embeddings is not None:
                # put embedding to search queue
                frame_id = save_frame(small_frame)
                notify_queue.put({
                    # "frame": small_frame,
                    "embeddings": embeddings,
                    "frame_id": frame_id
                })

        i = i + 1
        show_live_queue.put({
            "origin": True,
            "frame": roi_frame
        })
        time.sleep(interval)


if __name__ == '__main__':
    funcs = {
        "start_stream": start_stream,
        "search_and_notify": search_and_notify,
        "show_related_im": show_related_im,
        "show_live_frames": show_live_frames
    }

    rs = list()
    for name, func in funcs.items():
        t = threading.Thread(target=func)
        t.name = name
        t.start()
        rs.append(t)

    for t in rs:
        t.join()
