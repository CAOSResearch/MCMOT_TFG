from ultralytics import YOLO

def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    return x1, y1, x2, y2

def run_inference(model, source, tracker_cfg):

    results_stream = model.track(
        source=source,
        tracker=tracker_cfg,
        stream=True,
        persist=True,
        conf=0.4,
        iou=0.5,
        vid_stride=1
    )

    frame_id = 0

    for r in results_stream:
        frame = r.orig_img
        h, w = frame.shape[:2]

        detections = []

        if r.boxes is None:
            frame_id += 1
            continue

        for b in r.boxes:

            if b.id is None:
                continue

            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "track_id": int(b.id),
                "frame_id": frame_id,
                "timestamp": frame_id
            })

        yield frame, detections

        frame_id += 1