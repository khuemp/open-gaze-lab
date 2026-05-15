"""Shared helpers used by both the DD and GiW head-mounted loaders."""


def extract_video_metadata(video_path: str) -> dict:
    """Read fps / resolution / duration from a video file via OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "duration_s": n_frames / fps if fps > 0 else 0.0,
            "n_frames": n_frames,
        }
    finally:
        cap.release()
