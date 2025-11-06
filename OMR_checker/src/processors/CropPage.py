"""
https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
"""
import cv2
import numpy as np

from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils

MIN_PAGE_AREA = 80000


def normalize(image):
    return cv2.normalize(image, 0, 255, norm_type=cv2.NORM_MINMAX)


def check_max_cosine(approx):
    # assumes 4 pts present
    max_cosine = 0
    min_cosine = 1.5
    for i in range(2, 5):
        cosine = abs(angle(approx[i % 4], approx[i - 2], approx[i - 1]))
        max_cosine = max(cosine, max_cosine)
        min_cosine = min(cosine, min_cosine)

    if max_cosine >= 0.35:
        logger.warning("Quadrilateral is not a rectangle.")
        return False
    return True


def validate_rect(approx):
    return len(approx) == 4 and check_max_cosine(approx.reshape(4, 2))


def angle(p_1, p_2, p_0):
    dx1 = float(p_1[0] - p_0[0])
    dy1 = float(p_1[1] - p_0[1])
    dx2 = float(p_2[0] - p_0[0])
    dy2 = float(p_2[1] - p_0[1])
    return (dx1 * dx2 + dy1 * dy2) / np.sqrt(
        (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
    )


class CropPage(ImagePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cropping_ops = self.options
        self.morph_kernel = tuple(
            int(x) for x in cropping_ops.get("morphKernel", [5, 5]) # Used to be 10,10
        )

    def apply_filter(self, image, file_path):
        print("Entered CropPage preprocessor")
        image_mine = image.copy()
        image = normalize(cv2.GaussianBlur(image, (3, 3), 0))

        # Resize should be done with another preprocessor is needed
        sheet = self.find_page(image, file_path)
        if len(sheet) == 0:
            logger.error(
                f"\tError: Paper boundary not found for: '{file_path}'\nHave you accidentally included CropPage preprocessor?"
            )
            return None

        logger.info(f"Found page corners: \t {sheet.tolist()}")

        cv2.imshow("Image of OMR after croppage, before 4 point transform", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Warp layer 1
        image = ImageUtils.four_point_transform(image, sheet)

        # Having an unblurred version of the image and returning that instead so that QR code would be read fine
        image_mine = ImageUtils.four_point_transform(image_mine, sheet)

        cv2.imshow("Image of OMR after croppage, and after 4 point transform", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imshow("Image of MY_OMR (no blur) after croppage, and after 4 point transform", image_mine)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # cv2.imwrite('blur_and_crop.jpg', image)
        # cv2.imwrite('no_blur_and_crop.jpg', image_mine)

        # Return preprocessed image
        # return image
        # h, w = image_mine.shape[:2]
        # target_ratio = 1.414
        # current_ratio = h / w
        # if abs(current_ratio - target_ratio) > 0.01:
        #     new_h = int(w * target_ratio)
        #     image_mine = cv2.resize(image_mine, (w, new_h))

        # Returning the cropped but unblurred image
        return image_mine   # Toggle between blurred "image" and non-blurred "image_mine"

    def find_page(self, image, file_path):
        config = self.tuning_config

        image = normalize(image)

        _ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_TRUNC)
        image = normalize(image)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)

        # Close the small holes, i.e. Complete the edges on canny image
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        edge = cv2.Canny(closed, 55, 185) # 185, 55 before

        if config.outputs.show_image_level >= 5:
            InteractionUtils.show("edge", edge, config=config)

        # findContours returns outer boundaries in CW and inner ones, ACW.
        cnts = ImageUtils.grab_contours(
            cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )
        # convexHull to resolve disordered curves due to noise
        cnts = [cv2.convexHull(c) for c in cnts]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        sheet = []
        for c in cnts:
            if cv2.contourArea(c) < MIN_PAGE_AREA:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon=0.01 * peri, closed=True) # Used to be 0.025
            if validate_rect(approx):
                sheet = np.reshape(approx, (4, -1))
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
                cv2.drawContours(edge, [approx], -1, (255, 255, 255), 10)
                break

        return sheet
