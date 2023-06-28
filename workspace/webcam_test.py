from pkg_resources import resource_filename
from openvino.runtime import Core
import numpy as np
import cv2

from utils import non_max_suppression, put_text_on_image, draw_boxes_with_scores


import click
import time
import os

from deep_sort_realtime.deepsort_tracker import DeepSort

class FectureExtractor:
    def __init__(self, model=resource_filename(__name__, 'model/backbone-mobilenetv3-large/backbone-mobilenetv3-large.xml'), device='CPU'):
        # load and compile the model
        ie = Core()
        model = ie.read_model(model=model)
        compiled_model = ie.compile_model(model=model, device_name=device)
        self.model = compiled_model

        self.output_embeds = self.model.output(0)

    def preprocess(self, image):
        """
            image: RGB numpy array [H, W, C]
        """
        # ?
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])

        # Resize to [256, 256]
        input_tensor = cv2.resize(image, dsize=[256,256])

        # Add batch dim
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Convert to [0, 1] float32
        input_tensor = input_tensor.astype('float32') / 255.0

        # Change to channels first
        input_tensor = np.transpose(input_tensor, axes=(0, 3, 1, 2))

        return input_tensor

    def posprocess(self, model_out):
        return model_out

    def inference(self, image):
        input_tensor = self.preprocess(image)
        # output_tensor = self.model.inference(input_tensor)
        output_tensor = self.model( [input_tensor] )[self.output_embeds]
        embedding = self.posprocess(output_tensor)
        return embedding

class FaceDetector:
    model = None
    _cache_path = resource_filename(__name__, './model/cache')
    def __init__(self, model, device='CPU', confidence_thr=0.5, overlap_thr=0.7):
        if self.model == None:
            os.makedirs(self._cache_path, exist_ok=True)
            enable_caching = True
            config_dict = {"CACHE_DIR": str(self._cache_path)} if enable_caching else {}

            # load and compile the model
            ie = Core()
            model = ie.read_model(model=model)
            compiled_model = ie.compile_model(model=model, device_name=device, config=config_dict)
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

@click.command()
@click.option('-v','--video', default='/dev/video0')
@click.option('-c','--confidence', type=float, default=0.7)
@click.option('-m','--model', default=resource_filename(__name__, 'model/ultra-lightweight-face-detection-slim-320.xml'))
def main(video, model, confidence):

    encoder = FectureExtractor()
    detector = FaceDetector(model, device='CPU', confidence_thr=confidence, overlap_thr=0.7)
    video = cv2.VideoCapture(video)

    tracker = DeepSort(max_age=60, embedder=None)

    n_frames = 0
    fps_cum = 0.0
    fps_avg = 0.0
    while True:
        ret, frame = video.read()
        assert ret == True, 'Failed to video source of end of the file'

        start_time = time.perf_counter()
        faces, scores = detector.inference(frame)
        end_time = time.perf_counter()

        n_frames += 1
        fps = 1.0 / (end_time - start_time)
        fps_cum += fps
        fps_avg = fps_cum / n_frames

        frame = draw_boxes_with_scores(frame, faces, scores)
        frame = put_text_on_image(frame, text='FPS: {:.2f}'.format( fps_avg ))

        if len(faces) != 0:
            # bbs = [ (*bbox, scores[i], 'face') for i, bbox in enumerate(faces)]
            bbs = []
            embeds = []
            for i, bbox in enumerate(faces):
                # print(bbox, scores[i][0], 'face')
                xmin, ymin, xmax, ymax = bbox
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                bbs.append( (bbox, scores[i][0], 'face') )

                embedding = encoder.inference( frame[ymin:ymax, xmin:xmax,:] )
                embeds.append( embedding[0] )
                print(embedding.shape)

            tracks = tracker.update_tracks(bbs,
                                           frame=frame,
                                           embeds=embeds) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                # print('[DEBUG] track_id', track_id)
                # print('[DEBUG] ltrb', ltrb)

                x_min, y_min, x_max, y_max = ltrb
                cv2.rectangle(frame,
                              (int(x_min), int(y_min)),
                              (int(x_max), int(y_max)),
                              color=(255,0,0),
                              thickness=2)
                cv2.putText(frame,
                            str(track_id),
                            (int(x_max) + 10, int(y_max)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255,0,0),
                            1,
                            cv2.LINE_AA)

        cv2.imshow('webcam', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

if __name__ == '__main__':
    main()