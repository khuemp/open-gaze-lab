import base64


MIME_TYPES = {
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'webp': 'image/webp',
    'bmp': 'image/bmp',
}


def encode_image_bytes(data, ext):
    """Encode raw image bytes to a base64 data URI for Plotly.

    Args:
        data: The image file's bytes.
        ext: File extension with or without a leading dot ('png', '.jpg', ...).
            Unrecognized extensions fall back to ``image/png``.

    Returns:
        A ``data:<mime>;base64,...`` URI string.
    """
    mime_type = MIME_TYPES.get(ext.lower().lstrip('.'), 'image/png')
    encoded = base64.b64encode(data).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"


def encode_image_base64(image_path):
    """Encode an image file to a base64 data URI for Plotly."""
    ext = image_path.lower().split('.')[-1]
    with open(image_path, 'rb') as f:
        return encode_image_bytes(f.read(), ext)


def resolve_image_source(value):
    """Accept either a filesystem path or an already-built data URI.

    Lets callers that never have the image on disk — the browser build, where
    the upload arrives as bytes — pass ``encode_image_bytes(...)`` output
    straight through to the plotters.
    """
    if isinstance(value, str) and value.startswith('data:'):
        return value
    return encode_image_base64(value)
