from ultralytics import YOLO
import supervision as sv
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import read_stub, save_stubs


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5)
            detections += batch_detections
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            print("Loaded player tracks from stub")
            return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            player_class_id = cls_names_inv.get("Player", None)
            if player_class_id is not None:
                keep = detection_supervision.class_id == player_class_id
                detection_supervision = detection_supervision[keep]
                detection_supervision = detection_supervision[detection_supervision.confidence > 0.5]
            detection_supervision = detection_supervision.with_nms(threshold=0.4)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks.append({})

            for i in range(len(detection_with_tracks)):
                bbox = detection_with_tracks.xyxy[i].tolist()
                track_id = detection_with_tracks.tracker_id[i]

                if track_id is None:
                    continue

                tracks[frame_num][int(track_id)] = {"bbox": bbox}

        save_stubs(stub_path, tracks)
        print("Saved player tracks to stub")
        return tracks
