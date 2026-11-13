from py_reform import straighten
import cv2
import numpy as np
from deskew import determine_skew
from py_reform import straighten


## Correct skew in full image ##
def correct_skew(image):

    # Grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    # Estimate skew angle
    angle = determine_skew(gray)

    # Handle cases where no angle is detected
    if angle is None:
        return image.copy()

    # Avoid unnecessary rotation
    if abs(angle) < 0.5:
        return image.copy()

    h, w = gray.shape[:2]

    rotation_matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0
    )

    # Rotate the original crop
    corrected = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return corrected

## Correct warp in full image ##
def correct_warp(image_path):

    try:
        # Correct image warp
        corrected = straighten(
            image_path,
            model="uvdoc",
            device="cpu"
        )

        # Convert to OpenCV image
        corrected = np.array(
            corrected.convert("RGB")
        )

        corrected = cv2.cvtColor(
            corrected,
            cv2.COLOR_RGB2BGR
        )

        return corrected

    except Exception as e:
        print(f"Warp correction failed: {e}")
        return cv2.imread(image_path)