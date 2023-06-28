import time
import click

import cv2

from tracker import Tracker
from face_detector import FaceDetector
from utils import put_text_on_image, draw_boxes_with_scores

@click.command()
@click.option('-v','--video', default=0)
@click.option('-c','--confidence', type=float, default=0.7)
def main(video, confidence):

    detector = FaceDetector(device='CPU', confidence_thr=confidence, overlap_thr=0.7)
    tracker = Tracker(max_age=30,
                      n_init=3,
                      max_cosine_distance= 0.2
                      )
    video = cv2.VideoCapture(video)

    n_frames = 0
    fps_cum = 0.0
    fps_avg = 0.0
    while True:
        ret, frame = video.read()
        assert ret == True, 'Failed to video source of end of the file'

        start_time = time.perf_counter()
        bboxes, scores = detector.inference(frame)
        end_time = time.perf_counter()

        n_frames += 1
        fps = 1.0 / (end_time - start_time)
        fps_cum += fps
        fps_avg = fps_cum / n_frames

        frame = draw_boxes_with_scores(frame, bboxes, scores)
        frame = put_text_on_image(frame, text='FPS: {:.2f}'.format( fps_avg ))

        if len(bboxes) != 0:

            clss_ids = [0] * len(bboxes)
            tracks = tracker.update(frame, bboxes, scores, clss_ids)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()

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