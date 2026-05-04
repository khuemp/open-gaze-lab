import base64


def encode_image_base64(image_path):
    """Encode an image file to a base64 data URI for Plotly."""
    ext = image_path.lower().split('.')[-1]
    mime_types = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                  'gif': 'image/gif', 'webp': 'image/webp', 'bmp': 'image/bmp'}
    mime_type = mime_types.get(ext, 'image/png')

    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"
