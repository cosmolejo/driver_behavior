
import cv2
import logging
from pathlib import Path
from typing import Any


def slice_frame(video_folder, sample_dict):
    """Lee window_size frames consecutivos desde frame_start."""
    cap = None
    try:
        def _scalar(v):
            return v[0] if isinstance(v, list) else v

        start = int(_scalar(sample_dict['frame_start']))
        end = int(_scalar(sample_dict['frame_end']))
        file_id = str(_scalar(sample_dict['file_id'])).strip()
        if not file_id:
            return []

        num_needed = end - start
        if num_needed <= 0:
            return []

        video_path = str(Path(video_folder) / f"{file_id}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning("No se pudo abrir: %s", video_path)
            return []

        # Recortar al final del video si hace falta (sin shift hacia atrás)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            end = min(end, total)
            start = max(0, min(start, end))
            num_needed = end - start
            if num_needed <= 0:
                return []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(num_needed):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    except Exception:
        logging.exception("slice_frame falló: %r", sample_dict)
        return []
    finally:
        if cap is not None:
            cap.release()