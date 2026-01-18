# COLOR PROCESSING
# Reads the color card and finds the appropriate color adjustment matrix for an image.

import colour
from colour_checker_detection import detect_color_checkers_segmentation

image = "colortest1.jpg"

# 1. Detect the card (it finds the boundary regardless of patch count)
checkers = detect_color_checkers_segmentation(image)

# 2. Identify the card type automatically
# The library can compare the detected layout against its library 
# of 'ColorChecker 24', 'SpyderCheckr 24', 'Digital ColorChecker SG', etc.
for checker in checkers:
    # It returns the samples in a standardized order 
    # regardless of how the card was rotated in the photo.
    sampled_colors = checker.sampled_values 
    print("Colors sampled: " + sampled_colors)
    
    # 3. Apply the correction using the specific reference for that card
    # (Matches 24-patch samples to 24-patch targets automatically)