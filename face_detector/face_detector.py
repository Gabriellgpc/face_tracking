from pkg_resources import resource_filename
from openvino.runtime import Core
import numpy as np
import cv2

from utils import non_max_suppression


class FaceDetector:
    model = None
    def __init__(self,
                 model=resource_filename(__name__, 'ultra-lightweight-face-detection-slim-320/ultra-lightweight-face-detection-slim-320.xml'),
                 device='CPU',
                 confidence_thr=0.5,
                 overlap_thr=0.7):
        if self.model == None:
            # load and compile the model
            ie = Core()
            model = ie.read_model(model=model)
            compiled_model = ie.compile_model(model=model, device_name=device)
            self.model = compiled_model

        self.output_scores_layer = self.model.output(0)
        self.output_boxes_layer  = self.model.output(1)
        self.confidence_thr = confidence_thr
        self.overlap_thr = overlap_thr

    def preprocess(self, image):
        input_image = cv2.resize(image, dsize=[320,240])
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image.transpose(2,0,1), axis=0)
        return input_image

    def posprocess(self, pred_scores, pred_boxes, image_shape):
        # get all predictions with more than confidence_thr of confidence
        filtered_indexes = np.argwhere( pred_scores[0,:,1] > self.confidence_thr  ).tolist()
        filtered_boxes   = pred_boxes[0,filtered_indexes,:]
        filtered_scores  = pred_scores[0,filtered_indexes,1]

        if len(filtered_scores) == 0:
            return [],[]

        # convert all boxes to image coordinates
        h, w = image_shape
        def _convert_bbox_format(*args):
            bbox = args[0]
            x_min, y_min, x_max, y_max = bbox
            x_min = int(w*x_min)
            y_min = int(h*y_min)
            x_max = int(w*x_max)
            y_max = int(h*y_max)
            return x_min, y_min, x_max, y_max

        bboxes_image_coord = np.apply_along_axis(_convert_bbox_format, axis = 2, arr=filtered_boxes)

        # apply non-maximum supressions
        bboxes_image_coord, indexes = non_max_suppression(bboxes_image_coord.reshape([-1,4]), overlapThresh=self.overlap_thr)
        return bboxes_image_coord, filtered_scores[indexes]

    def draw_bboxes(self, image, bboxes, color=[0,255,0]):
        # draw all bboxes on the input image
        for boxe in bboxes:
            x_min, y_min, x_max, y_max = boxe
            pt1 = (x_min, y_min)
            pt2 = (x_max, y_max)
            cv2.rectangle(image, pt1, pt2, color=color, thickness=2, lineType=cv2.LINE_4)#BGR

    def inference(self, image):
        input_image = self.preprocess(image)
        # inference
        pred_scores = self.model( [input_image] )[self.output_scores_layer]
        pred_boxes = self.model( [input_image] )[self.output_boxes_layer]

        image_shape = image.shape[:2]
        faces, scores = self.posprocess(pred_scores, pred_boxes, image_shape)
        return faces, scores