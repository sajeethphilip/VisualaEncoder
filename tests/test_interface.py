import cv2
import numpy as np
from interface.region_marking import mark_region

def test_region_marking():
    """Test the region marking functionality."""
    # Create a dummy image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Simulate marking a region (this will require manual input in a real test)
    roi = mark_region(image)
    
    # Check if ROI is a tuple of 4 values (x, y, width, height)
    assert len(roi) == 4, "ROI should have 4 values"
