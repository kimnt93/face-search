import copy
import logging
import face_recognition
import cv2
import torch
from facelib import FaceDetector, FaceRecognizer
from facelib import get_config
import numpy
from lib import config
from numpy import linalg as LA

__all__ = ["FaceRecognitionWithMask", "FaceRecognitionSimple"]

LOGGER = logging.getLogger()
LOGGER.setLevel("DEBUG")


class FaceDetectionModel:
    def __init__(self, scale_ratio=0.25):
        self.scale_ratio = scale_ratio
        self.scale_size = 1 / self.scale_ratio

    def get_faces(self, image):
        raise NotImplementedError


class FaceRecognitionWithMask(FaceDetectionModel):
    def __init__(self, scale_ratio=0.25):
        super().__init__(scale_ratio)
        self.CROP_FACE_SHAPE = (112, 112)
        conf = get_config()
        conf.use_mobilfacenet = True
        self.recognizer = FaceRecognizer(conf, verbose=True)
        self.recognizer.model.eval()
        self.detector = FaceDetector(name='mobilenet', weight_path='models/rtn_mobilenet.pth', device='cpu')

    def get_faces(self, image):
        # resize image
        small_frame = cv2.resize(image, (0, 0), fx=self.scale_ratio, fy=self.scale_ratio)
        rgb_small_frame = small_frame[:, :, ::-1]
        faces, face_boxes, _, _ = self.detector.detect_align(small_frame)

        if faces.nelement() > 4:
            LOGGER.debug(f"Detect faces : {face_boxes}")
            embeddings = self.recognizer.feature_extractor(faces).detach().numpy()
            boxes = face_boxes * self.scale_size
            for top, right, bottom, left in boxes.tolist():
                # Draw a box around the face
                x1, y1, x2, y2 = int(top), int(right), int(bottom), int(left)
                # Draw a box around the face
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            embedding = embeddings[0]
            embedding = numpy.array([embedding])
            embedding = embedding / LA.norm(embedding)
            face_boxes = [int(x) for x in face_boxes[0]]
            small_frame = crop_and_extend(
                small_frame,
                [face_boxes[0], face_boxes[3], face_boxes[1], face_boxes[2]],
                delta=78, target_size=(120, 120)
            )
            return image, small_frame, embedding

        return image, small_frame, None


def crop_and_extend(image, bbox, delta, target_size):
    top, right, bottom, left = bbox
    height, width = image.shape[:2]

    # Extend the bounding box
    top = max(0, top - delta)
    right = min(width, right + delta)
    bottom = min(height, bottom + delta)
    left = max(0, left - delta)

    # Crop the extended bounding box from the image
    cropped_face = image[top:bottom, left:right]

    # Resize the cropped face to the target size
    cropped_face_resized = cv2.resize(cropped_face, target_size)

    return cropped_face_resized


def get_largest_box(face_boxes):
    if len(face_boxes) == 1:
        return face_boxes[0]
    elif len(face_boxes) == 0:
        return None
    else:
        areas = [(box[1] - box[3]) * (box[2] - box[0]) for box in face_boxes]
        largest_index = areas.index(max(areas))

        # Get the largest box
        largest_box = face_boxes[largest_index]
        return largest_box




class FaceRecognitionSimple(FaceDetectionModel):
    def get_faces(self, image):
        small_frame = cv2.resize(image, (0, 0), fx=self.scale_ratio, fy=self.scale_ratio)
        rgb_small_frame = small_frame
        # rgb_small_frame = small_frame[:, :, ::-1]
        face_boxes = face_recognition.face_locations(rgb_small_frame)  # return array of tuple (boxes)
        face_box = get_largest_box(face_boxes)

        if face_box:
            embeddings = face_recognition.face_encodings(
                rgb_small_frame,
                [face_box],
                model='large'
            )
            embedding = embeddings[0]

            top, right, bottom, left = face_box

            # upscale
            top, right, bottom, left = int(top * self.scale_size), int(right * self.scale_size), int(bottom * self.scale_size), int(left * self.scale_size)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            LOGGER.debug(f"Detect faces : {face_box}")
            embedding = numpy.array([embedding])
            small_frame = crop_and_extend(small_frame, face_box, delta=78, target_size=(120, 120))
            return image, small_frame, embedding
        else:
            return image, small_frame, None
