import cv2

def mark_region(image):
    """Allow the user to mark a region of interest on the image."""
    marked_image = image.copy()
    roi = cv2.selectROI("Mark Region", marked_image)
    cv2.destroyAllWindows()
    return roi
