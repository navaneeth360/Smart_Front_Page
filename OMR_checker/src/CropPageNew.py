import cv2
import numpy as np

from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils

import docscan.doc as scan

A4_RATIO = 1.414  # height / width for A4 sheet

class CropPage(ImagePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cropping_ops = self.options
        self.morph_kernel = tuple(
            int(x) for x in cropping_ops.get("morphKernel", [10, 10])
        )

    def apply_filter(self, image, file_path):
        """
        Main entry: finds the page, warps it, and resizes to A4 ratio.
        """
        orig = image.copy()

        try:
            # encode image as bytes since docscan expects raw bytes
            success, encoded = cv2.imencode('.png', image)
            if not success:
                logger.error(f"Encoding failed for {file_path}")
                return None
            data = encoded.tobytes()

            # call docscan
            result_bytes = scan(data)
            # convert bytes back to numpy image
            nparr = np.frombuffer(result_bytes, np.uint8)
            cropped = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if cropped is None:
                logger.error(f"docscan failed to crop {file_path}")
                return None

            # enforce A4 ratio
            h, w = cropped.shape[:2]
            target_w = 1000
            target_h = int(target_w * A4_RATIO)
            output = cv2.resize(cropped, (target_w, target_h))
            print("Output shape is ", cropped.shape)
            return cropped

        except Exception as e:
            logger.error(f"Error in CropPage for {file_path}: {e}")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = self.find_page(gray)

        if corners is None:
            logger.error(f"Page not found for {file_path}")
            return None

        # Order points and warp
        ordered = self.order_points(corners)
        warped = self.four_point_transform(orig, ordered)

        # Enforce A4 aspect ratio
        h, w = warped.shape[:2]
        desired_w = 1000
        desired_h = int(desired_w * A4_RATIO)
        resized = cv2.resize(warped, (desired_w, desired_h))

        return resized

    def find_page(self, gray):
        """
        Detects the page corners using Hough lines and intersections.
        """
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Detect straight lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=200, maxLineGap=20)
        if lines is None:
            return None

        lines = lines[:, 0, :]  # simplify shape
        horiz = [l for l in lines if abs(l[1] - l[3]) > abs(l[0] - l[2])]
        vert = [l for l in lines if abs(l[0] - l[2]) >= abs(l[1] - l[3])]

        if len(horiz) < 2 or len(vert) < 2:
            return None

        # Get outermost lines (min/max)
        top = min(horiz, key=lambda l: min(l[1], l[3]))
        bottom = max(horiz, key=lambda l: max(l[1], l[3]))
        left = min(vert, key=lambda l: min(l[0], l[2]))
        right = max(vert, key=lambda l: max(l[0], l[2]))

        # Compute intersections of these 4 lines
        corners = []
        for h_line in [top, bottom]:
            for v_line in [left, right]:
                inter = self.line_intersection(h_line, v_line)
                if inter is not None:
                    corners.append(inter)

        if len(corners) != 4:
            return None
        return np.array(corners, dtype="float32")

    def line_intersection(self, line1, line2):
        """
        Returns intersection point of two lines given by endpoints.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1

        A2 = y4 - y3
        B2 = x3 - x4
        C2 = A2 * x3 + B2 * y3

        det = A1 * B2 - A2 * B1
        if abs(det) < 1e-10:
            return None

        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return [x, y]

    def order_points(self, pts):
        """
        Orders 4 points as TL, TR, BR, BL.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def four_point_transform(self, image, pts):
        """
        Performs a perspective transform to get a top-down view.
        """
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
