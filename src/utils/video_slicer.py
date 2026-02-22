
import cv2
import logging
from pathlib import Path
from typing import Any


def slice_frame(video_folder: str | Path, sample_dict: dict[str, Any]) -> list[Any]:
    """Create a list of frames from frame_start to frame_end (inclusive)."""
    cap: cv2.VideoCapture | None = None
    try:
        if not isinstance(sample_dict, dict):
            raise TypeError(f"sample_dict must be a dict, got {type(sample_dict)!r}")

        # Parse and validate inputs early to avoid confusing downstream errors.
        try:
            start_frame = int(sample_dict["frame_start"][0]) if type(sample_dict["frame_start"]) == list else int(sample_dict["frame_start"])
            end_frame = int(sample_dict["frame_end"][0]) if type(sample_dict["frame_end"]) == list else int(sample_dict["frame_end"])
        except KeyError as e:
            raise KeyError(f"Missing required key in sample_dict: {e.args[0]!r}") from e
        except (TypeError, ValueError) as e:
            raise ValueError("frame_start/frame_end must be int-convertible") from e

        file_id = str(sample_dict["file_id"][0]).strip() if type(sample_dict.get("file_id")) == list else str(sample_dict["file_id"]).strip()
        if not file_id:
            raise ValueError("sample_dict['file_id'] must be a non-empty value")

        video_path = str(Path(video_folder) / f"{file_id}.mp4")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning("Error opening video: %s", video_path)
            print('no se pudo abrir el video')
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            logging.warning("Video has no frames (or frame count unavailable): %s", video_path)
            return []

        # Clamp range to valid bounds.
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, total_frames - 1)

        if start_frame > end_frame:
            logging.warning("Invalid frame range for %s: %s-%s", file_id, start_frame, end_frame)
            return []

        sliced_frames: list[Any] = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cur = start_frame
        while cur <= end_frame:
            ret, frame = cap.read()
            if not ret:
                logging.warning(
                    "Failed to read frame %s (requested %s-%s) from %s",
                    cur,
                    start_frame,
                    end_frame,
                    video_path,
                )
                break
            sliced_frames.append(frame)
            cur += 1

        return sliced_frames

    except Exception:
        # Log full traceback; return a safe default to avoid None-propagation bugs.
        logging.exception("slice_frame failed for sample_dict=%r", sample_dict)
        return []
    finally:
        if cap is not None:
            cap.release()