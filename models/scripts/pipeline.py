import os
import cv2
from ultralytics import YOLO

from reid import ReIDExtractor
from global_id_manager import GlobalIDManager
from inference_engine import run_inference


def main():

    # Camera Configuration
    TRACKER_CFG = "custom_bytetrack.yaml"

    SOURCES = {
        0: "MCMOT_TFG/models/datasets/videos/video_7/video7_left.mp4",
        1: "MCMOT_TFG/models/datasets/videos/video_7/video7_right.mp4"
    }

    OUTPUT_DIR = "MCMOT_TFG/models/output/video_7"

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Model Selection
    models = {
        cam_id: YOLO(
            "MCMOT_TFG/models/scripts/runs/detect/multicamera_yolo11n_augmentation/weights/best.pt"
        )
        for cam_id in SOURCES
    }

    reid = ReIDExtractor()

    # Global ID Manager Parameter Configuration
    global_manager = GlobalIDManager(
        appearance_weight=0.30,
        temporal_weight=0.70,
        assignment_threshold=0.60,
        history_size=30,
        confirm_hits=3,
        expected_transition_frames=110,
        transition_sigma=40,
        max_transition_frames=160,
        lost_timeout=30,
        memory_timeout=900,
        short_occlusion_frames=60,
        short_occlusion_threshold=0.55
    )

    streams = {
        cam_id: run_inference(
            models[cam_id],
            SOURCES[cam_id],
            TRACKER_CFG
        )
        for cam_id in SOURCES
    }

    # Video Capture and Writer
    writers = {}

    for cam_id, src in SOURCES.items():

        cap = cv2.VideoCapture(src)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        cap.release()

        out_path = os.path.join(
            OUTPUT_DIR,
            f"camera_{cam_id}.mp4"
        )

        writers[cam_id] = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

    # Main
    frame_id = 0

    while True:

        frame_id += 1

        any_stream_alive = False

        for cam_id, stream in streams.items():

            try:
                frame, detections = next(stream)
                any_stream_alive = True

            except StopIteration:
                continue

            # ReID Extraction and Global ID Management
            valid_dets = []
            crops = []

            for det in detections:

                x1, y1, x2, y2 = det["bbox"]

                crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if crop.size == 0:
                    continue

                valid_dets.append(det)
                crops.append(crop)

            # ReID Extraction
            if len(crops) > 0:
                embeddings = reid.extract_batch(crops)
            else:
                embeddings = []

            # Detections processing
            for det, emb in zip(
                valid_dets,
                embeddings
            ):

                x1, y1, x2, y2 = det["bbox"]

                track_id = det["track_id"]

                gid = global_manager.add_observation(
                    frame_id=frame_id,
                    cam_id=cam_id,
                    track_id=track_id,
                    embedding=emb
                )

                # Output Visualization
                if gid is None:
                    color = (0, 255, 255)  # amarillo
                else:
                    color = (0, 255, 0)    # verde

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                if gid is not None:

                    cv2.putText(
                        frame,
                        f"GID {gid}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

            # Visual Display
            cv2.imshow(
                f"Camera {cam_id}",
                frame
            )

            writers[cam_id].write(
                frame
            )

        global_manager.cleanup(
            frame_id
        )


        if not any_stream_alive:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for writer in writers.values():
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()