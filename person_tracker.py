import os
from typing import IO
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import socket
from custom_socket import CustomSocket
import json

# config constants
SIZE = 416
YOLO_WEIGHTS_PATH = './checkpoints/yolov4-416'
IOU = 0.45
SCORE = 0.50
ENCODER_PATH = 'model_data/mars-small128.pb'

# custom allowed classes (uncomment line below to customize tracker for only people)
ALLOWED_CLASSES = ['person']

# # test
# saved_model_loaded = tf.saved_model.load(YOLO_WEIGHTS_PATH, tags=[tag_constants.SERVING])
# infer = saved_model_loaded.signatures['serving_default']

class PersonTracker:
    
    def __init__(self, max_cosine_distance = 0.4, nn_budget = None, nms_max_overlap = 1.0):

        # initialize deep sort
        self.encoder = gdet.create_box_encoder(ENCODER_PATH, batch_size=1)
        # calculate cosine distance metric
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        self.tracker = Tracker(self.metric)
        # load configuration for object detector
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.nms_max_overlap = nms_max_overlap
        self.saved_model_loaded = tf.saved_model.load(YOLO_WEIGHTS_PATH, tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']
        # read in all class names from config
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    def process(self, img2):

        # preprocess input img
        img = img2.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(img, (SIZE, SIZE))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # remem start time for calculating fps
        start_time = time.time()

        # run detections
        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)
        for _, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=IOU,
                score_threshold=SCORE
            )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = img.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = self.class_names[class_indx]
            if class_name not in ALLOWED_CLASSES:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = self.encoder(img, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # call the tracker
        self.tracker.predict()
        self.tracker.update(detections)


        # create solution list
        sol = []

        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: 
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # append to sol
            sol.append([track.track_id, class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))])


        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        result_img = np.asarray(img)
        result_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return sol, result_img


def main():

    # init model
    PT = PersonTracker()

    # load image
    img = cv2.imread('test_pics/test1.jpg')

    HOST = socket.gethostname()
    PORT = 11000

    # solution list, and result_img
    # sol, result_img = PT.process(img)

    # print(sol)

    server = CustomSocket(HOST,PORT)
    server.startServer()

    while True :
        conn, addr = server.sock.accept()
        print("Client connected from",addr)
        data = server.recvMsg(conn)
        img = np.frombuffer(data,dtype=np.uint8).reshape(720,1080,3)
        sol, result_img = PT.process(img)
        result = {}
        result["result"] = sol
        server.sendMsg(conn,json.dumps(result))

    # for i in range(1,8,1):
    #     img_path = 'C:/robocup2022/yolov4-deepsort/test_pics/test' + str(i) + '.jpg'
    #     img = cv2.imread(img_path)

    #     sol, result_img = PT.process(img)

    #     # display output
    #     print(sol)        
    #     cv2.imshow('Output', result_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

if __name__ == '__main__':
    main()