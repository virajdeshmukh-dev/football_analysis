from ultralytics import YOLO
import supervision as sv
from trackers import ByteTrackTracker
import pickle
import numpy as np
import cv2
import os
import sys
sys.path.append("../")
from football_analysis.utils import get_bbox_width, get_center_of_bbox


class Tracker:

    def __init__(self, model_path):
        # Initialise the detection model and tracking engine from the supplied path.
        print(f"[Tracker.__init__] Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.tracker = ByteTrackTracker()
        print("[Tracker.__init__] Model and tracker initialised")


    def detect_frames(self, frames):
        # Process frames in predictable batches before returning all detections.
        print(f"[Tracker.detect_frames] Starting detection for {len(frames)} frame(s)")
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            print(
                f"[Tracker.detect_frames] Predicting frames {i} "
                f"to {min(i + batch_size, len(frames)) - 1}"
            )
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detection_batch

        # Summarise the accumulated model output before returning it.
        print(f"[Tracker.detect_frames] Finished with {len(detections)} detection result(s)")
        return detections


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # Prefer an existing stub when requested so inference can be skipped.
        print(
            f"[Tracker.get_object_tracks] Starting for {len(frames)} frame(s); "
            f"read_from_stub={read_from_stub}, stub_path={stub_path}"
        )
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            print(
                "[Tracker.get_object_tracks] Loaded tracks from stub with "
                f"{len(tracks['players'])} frame(s)"
            )
            return tracks

        # Run object detection before converting detections into tracks.
        print("[Tracker.get_object_tracks] No usable stub found; running detection")
        detections = self.detect_frames(frames)

        tracks = {
                    'players':[],
                    'referees':[],
                    'ball':[]
                }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            #convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #convert goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] == cls_names_inv['player']

            #track objects

            detection_with_tracks = self.tracker.update(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            # Persist newly calculated tracks for future runs when a path is supplied.
            print(f"[Tracker.get_object_tracks] Saving tracks to stub: {stub_path}")
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        # Report the computed per-category frame totals before returning tracks.
        print(
            "[Tracker.get_object_tracks] Finished with "
            f"players={len(tracks['players'])}, referees={len(tracks['referees'])}, "
            f"ball={len(tracks['ball'])} frame entries"
        )
        return tracks

    def draw_triangle(self, frame, bbox, color, track_id):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # Derive ellipse geometry from the object's bounding box.
        print(f"[Tracker.draw_ellipse] Drawing track_id={track_id}, bbox={bbox}")
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width=40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) +15
        y2_rect = (y2 + rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(
                            frame,
                            (int(x1_rect), int(y1_rect)),
                            (int(x2_rect), int(y2_rect)),
                            color,
                            cv2.FILLED
                          )
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )


        # Return the annotated frame after the ellipse has been drawn.
        print(f"[Tracker.draw_ellipse] Finished drawing track_id={track_id}")
        return frame

    def draw_annotations(self, video_frames, tracks):
        # Copy and annotate each source frame before assembling the output video.
        print(f"[Tracker.draw_annotations] Annotating {len(video_frames)} frame(s)")
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            #draw players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0,0,225), track_id)

            #draw referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0,255,225), track_id)

            #draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0,255,0), track_id)

            output_video_frames.append(frame)

        # Report the annotation result before returning the completed frame list.
        print(
            "[Tracker.draw_annotations] Finished annotation with "
            f"{len(output_video_frames)} output frame(s)"
        )
        return output_video_frames
