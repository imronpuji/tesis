import cv2
import numpy as np
import os

# Define image path
image_path = './corns/Blight/Corn_Blight (676).JPG'

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: The image file does not exist at the path {image_path}")
else:
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image at path {image_path}")
    else:
        print("Image loaded successfully.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply thresholding
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Background area using dilation
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add 1 to all labels so that the background is 1 instead of 0
        markers = markers + 1

        # Mark the unknown regions with 0
        markers[unknown == 255] = 0

        # Apply the watershed algorithm
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]  # Mark boundaries in red

        # Save the segmented image
        output_path = './corns/Blight/segmented_image.jpeg'
        cv2.imwrite(output_path, image)
        print(f"Segmented image saved at {output_path}")

        # Optional: Show intermediate results for debugging
        # cv2.imshow('Threshold', thresh)
        # cv2.imshow('Opening', opening)
        # cv2.imshow('Distance Transform', dist_transform)
        # cv2.imshow('Sure Background', sure_bg)
        # cv2.imshow('Sure Foreground', sure_fg)
        # cv2.imshow('Markers', markers.astype(np.uint8) * 10)  # Display markers
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
