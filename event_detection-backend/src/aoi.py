"""
AOI (Area of Interest) detection module for eye tracking on text documents.

This module extracts text and creates bounding boxes for each word in images,
which can be used to determine which text a gaze point belongs to.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr


class AOI:
    """Area of Interest detection using OCR on image documents."""
    
    def __init__(self, languages: List[str] = None):
        """
        Initialize the AOI detector.
        
        Args:
            languages: List of languages for OCR (default: ['en'])
                      Examples: ['en'], ['en', 'de'], etc.
        """
        self.aoi_data = {}
        if languages is None:
            languages = ['en']
        # Initialize EasyOCR reader (downloads models on first use)
        self.reader = easyocr.Reader(languages, gpu=False)
    

    def extract_text_and_bboxes_from_image(self, image_path: str) -> Dict:
        """
        Extract text and bounding boxes from an image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing:
                - 'text': List of extracted words (individual words)
                - 'bboxes': List of bounding boxes in pixel coordinates (x0, y0, x1, y1) for each word
                - 'image_width': Original image width
                - 'image_height': Original image height
                - 'confidences': Confidence scores for each word (0-1)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Open image to get dimensions
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        # Run EasyOCR
        results = self.reader.readtext(image_path)
        
        words = []
        bboxes = []
        confidences = []
        
        # Process each detection
        for detection in results:
            # detection format: (bbox, text, confidence)
            # bbox is a list of 4 corner points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            bbox_points = detection[0]
            text = detection[1]
            confidence = detection[2]
            
            # Skip empty text
            if not text.strip():
                continue
            
            # Convert 4-point polygon to rectangular bbox (min/max)
            xs = [point[0] for point in bbox_points]
            ys = [point[1] for point in bbox_points]
            
            line_x0 = min(xs)
            line_y0 = min(ys)
            line_x1 = max(xs)
            line_y1 = max(ys)
            
            line_width = line_x1 - line_x0
            
            # Split text into words
            text_words = text.split()
            
            if len(text_words) == 1:
                # Single word - use original bbox
                words.append(text_words[0])
                bboxes.append([line_x0, line_y0, line_x1, line_y1])
                confidences.append(confidence)
            else:
                # Multiple words - distribute line width proportionally to each word
                text_without_spaces = text.replace(" ", "")
                char_width = line_width / len(text_without_spaces)
                current_x = line_x0
                
                for word in text_words:
                    word_width = char_width * len(word)
                    word_x0 = current_x
                    word_x1 = current_x + word_width
                    
                    words.append(word)
                    bboxes.append([word_x0, line_y0, word_x1, line_y1])
                    confidences.append(confidence)
                    
                    # Move to next word (NO extra space added)
                    current_x = word_x1  # <-- Changed: remove the "+ char_width"
            
        return {
            'text': words,
            'bboxes': bboxes,
            'image_width': width,
            'image_height': height,
            'confidences': confidences,
            'image_path': image_path
        }
    
    def process_aoi_dataset(self, dataset_path: str) -> Dict[str, Dict]:
        """
        Process all images in AOI dataset directory.
        
        Args:
            dataset_path: Path to the AOI_Dataset directory
            
        Returns:
            Dictionary mapping image filenames to their extracted data
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        results = {}
        
        # Process all image files (PNG, JPG, etc.)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        
        for image_file in dataset_path.rglob('*'):
            if image_file.suffix.lower() in image_extensions:
                data = self.extract_text_and_bboxes_from_image(str(image_file))
                results[image_file.name] = data
                self.aoi_data[image_file.name] = data        
        return results
    
    def get_word_at_gaze_point(self, gaze_point: Tuple[float, float], 
                               image_data: Dict) -> Optional[Dict]:
        """
        Determine which word a gaze point belongs to.
        
        Args:
            gaze_point: Gaze point in format (x, y) with pixel coordinates
            image_data: Image data dictionary from extract_text_and_bboxes_from_image
            
        Returns:
            Dictionary containing word information or None if not in any bounding box
        """
        gaze_x, gaze_y = gaze_point
        
        # Check each word's bounding box
        for i, bbox in enumerate(image_data['bboxes']):
            x0, y0, x1, y1 = bbox
            
            if x0 <= gaze_x <= x1 and y0 <= gaze_y <= y1:
                return {
                    'word': image_data['text'][i],
                    'word_index': i,
                    'bbox': bbox,
                    'confidence': image_data['confidences'][i]
                }
        
        return None
    
    def classify_gaze_points(self, gaze_points: List[Tuple[float, float]], 
                             image_data: Dict) -> List[Optional[Dict]]:
        """
        Classify multiple gaze points to their corresponding words.
        
        Args:
            gaze_points: List of gaze points in format [(x, y), ...]
            image_data: Image data dictionary
            
        Returns:
            List of word information for each gaze point (None if not in any bbox)
        """
        results = []
        for gaze_point in gaze_points:
            result = self.get_word_at_gaze_point(gaze_point, image_data)
            results.append(result)
        
        return results
    
    def visualize_bboxes(self, image_path: str, output_path: Optional[str] = None) -> Image.Image:
        """
        Visualize bounding boxes on the image.
        
        Args:
            image_path: Path to the image file
            output_path: Optional path to save the visualization
            
        Returns:
            PIL Image with bounding boxes drawn
        """
        from PIL import ImageDraw, ImageFont
        
        if image_path not in self.aoi_data:
            data = self.extract_text_and_bboxes_from_image(image_path)
        else:
            data = self.aoi_data[image_path]
        
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        width = data['image_width']
        height = data['image_height']
        
        # Draw bounding boxes and text labels
        for word, bbox in zip(data['text'], data['bboxes']):
            x0, y0, x1, y1 = bbox
            
            # Draw rectangle
            draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
            
            # Draw text label
            try:
                # Try to use a default font, fall back to default if not available
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((x0, y0 - 10), word, fill='blue', font=font)
        
        if output_path:
            image.save(output_path)
        
        return image


def main():
    """Example usage of the AOI module - test bounding box visualization."""
    
    # Initialize AOI detector
    aoi = AOI()
    
    # Path to AOI dataset
    dataset_base_path = Path(__file__).parent.parent.parent / 'data' / 'Eye_Tracking_Datasets'
    text_aoi_path = dataset_base_path / 'Text' / 'AOI_Dataset'
    
    # Create output directory
    output_dir = dataset_base_path / 'Text' / 'visualized_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find first image in dataset
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    image_files = [f for f in text_aoi_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    for image_file in image_files:
        image_path = str(image_file)
        try:
            data = aoi.extract_text_and_bboxes_from_image(image_path)
            output_path = output_dir / f"visualized_{image_file.name}"
            aoi.visualize_bboxes(image_path, str(output_path))
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    main()
