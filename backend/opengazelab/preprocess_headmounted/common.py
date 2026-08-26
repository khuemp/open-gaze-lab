"""Shared helpers used by both the DD and GiW head-mounted loaders.

Video metadata has two sources depending on where the pipeline runs:

* **Natively** (notebook / CLI) OpenCV reads it straight off the file.
* **In the browser** OpenCV's wasm build ships no video codecs and cannot
  demux an MP4 at all, so the host page probes the container itself (see
  ``frontend/src/video/probeVideo.js``) and hands the result to
  :func:`register_video_metadata` before invoking a loader.

Registered metadata always wins, which also makes the loaders easy to test
without a real video file.
"""

_METADATA_KEYS = ("fps", "width", "height", "duration_s", "n_frames")

_INJECTED: dict = {}


def register_video_metadata(video_path: str, meta: dict) -> dict:
    """Pre-supply the metadata :func:`extract_video_metadata` should return.

    Args:
        video_path: Path the loaders will be called with — must match exactly.
        meta: Dict with ``fps``, ``width``, ``height``, ``duration_s`` and
            ``n_frames``. Missing keys default to 0.

    Returns:
        The normalized dict that was stored.
    """
    normalized = {key: meta.get(key, 0) for key in _METADATA_KEYS}
    normalized["fps"] = float(normalized["fps"] or 0.0)
    normalized["width"] = int(normalized["width"] or 0)
    normalized["height"] = int(normalized["height"] or 0)
    normalized["duration_s"] = float(normalized["duration_s"] or 0.0)
    normalized["n_frames"] = int(normalized["n_frames"] or 0)

    _INJECTED[str(video_path)] = normalized
    return normalized


def clear_video_metadata(video_path: str = None) -> None:
    """Forget metadata for *video_path*, or all of it when called bare."""
    if video_path is None:
        _INJECTED.clear()
    else:
        _INJECTED.pop(str(video_path), None)


def extract_video_metadata(video_path: str) -> dict:
    """Read fps / resolution / duration for a video.

    Returns metadata registered via :func:`register_video_metadata` when
    present, otherwise falls back to reading the file with OpenCV.
    """
    registered = _INJECTED.get(str(video_path))
    if registered is not None:
        return dict(registered)

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"No metadata registered for {video_path!r} and OpenCV is unavailable. "
            "Call register_video_metadata() first when running without cv2 "
            "(e.g. in Pyodide)."
        ) from exc

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
