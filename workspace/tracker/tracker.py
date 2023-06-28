from deep_sort_realtime.deepsort_tracker import DeepSort

from pkg_resources import resource_filename
from openvino.runtime import Core
import numpy as np
import cv2

class FectureExtractor:
    def __init__(self, model=resource_filename(__name__, 'backbone-mobilenetv3-large/backbone-mobilenetv3-large.xml'), device='CPU'):
        # load and compile the model
        ie = Core()
        model = ie.read_model(model=model)
        compiled_model = ie.compile_model(model=model, device_name=device)
        self.model = compiled_model
        self.output_embeds = self.model.output(0)

    def preprocess(self, image):
        """
            image: BGR numpy array [H, W, C]
        """

        # Resize to [256, 256]
        input_tensor = cv2.resize(image, dsize=[256,256])

        # Convert from BGR to RGB format
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)

        # Add batch dim
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Convert to [0, 1] float32
        input_tensor = input_tensor.astype('float32') / 255.0

        # Normalize to imagenet statics
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        input_tensor = (input_tensor - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

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

class Tracker:
    def __init__(self,
                 max_iou_distance: float = 0.7,
                 max_age: int = 30,
                 n_init: int = 3,
                 nms_max_overlap: float = 1,
                 max_cosine_distance: float = 0.2,
                 ):
        """
            Parameters:
            ----------
                max_iou_distance:
                    Maximum iou distance between previous and current bboxes.
                    if that distance exceeds max_iou_distance, then count as a missed object/new one.
                max_age:
                    Maximum age to wait until considering a missed object
                n_init:
                    number of detects to start tracking an object
                nms_max_overlap:
                    maximum overlap between detections
                max_cosine_distance:
                    distance between embeddings of objects. That parameters helps to controls the functions responsavel to reidentify objects.
                    greater means the object should appear more similar to the previous detections to considerer the reidentification.           
        """
        self.tracker_deepsort = DeepSort(max_age=max_age,
                                         n_init=n_init,
                                         max_iou_distance=max_iou_distance,
                                         nms_max_overlap=nms_max_overlap,
                                         max_cosine_distance=max_cosine_distance,
                                         embedder=None,
                                         )
        self.embedder = FectureExtractor()

    def update(self, frame, bboxes, scores, class_ids):
        """
            bboxes:
                list of bboxes: [xmin, ymin, xmax, ymax]. int
            scores:
                list of scores/confidences for each detection, float
            class_ids:
                list of class ids for each detection, int
            frame:
                An BGR
        """

        # detections:
        #     expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

        # Loop through each detection and extract the region of the objects to passthrough the embedder network
        # embeds = [ self.embedder.inference(frame[det[0][],,:])[0] for det in detections]
        embeddings = []
        detections = []
        for i, cls_id in enumerate(class_ids):
            xmin, ymin, xmax, ymax = bboxes[i]

            width  = xmax - xmin
            height = ymax - ymin
            # build detection list in the format expected by deepSORT
            detection = [ [xmin, ymin, width, height], scores[i], cls_id]

            # Compute embedding for the detected object
            object_isolated = frame[ymin:ymax, xmin:xmax, :]
            embedding = self.embedder.inference( object_isolated )

            # append to the detections and embeddings list
            detections.append( detection )
            embeddings.append( embedding[0] )

        tracks = self.tracker_deepsort.update_tracks(detections,
                                                     frame=frame,
                                                     embeds=embeddings)
        # if not track.is_confirmed():
        #     continue
        # track_id = track.track_id
        # ltrb = track.to_ltrb()
        return tracks