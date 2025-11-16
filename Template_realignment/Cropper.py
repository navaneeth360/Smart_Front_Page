"""
https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
"""
import cv2
import numpy as np
import argparse 
import os       

# --- DEFAULT PATHS FOR NO-ARGUMENT RUNS ---
# CHANGE THESE TO MATCH YOUR SYSTEM FOR EASY TESTING!
DEFAULT_INPUT_PATH = r"C:\Users\Navaneeth\BTP_1\Final_repo\inputs\WhatsApp Image 2025-11-02 at 00.15.05.jpeg" #r"C:\Users\Navaneeth\BTP_1\OMRChecker\inputs\Template_sizing\t1\t1_p1.jpeg"
DEFAULT_OUTPUT_DIR = r"C:\Users\Navaneeth\BTP_1\Final_repo\outputs\cropped"
# ------------------------------------------

MIN_PAGE_AREA = 80000

def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        max_width = max(int(width_a), int(width_b))
        # max_width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

        # compute the height of the new image, which will be the
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

        # return the warped image
        return warped

def grab_contours(cnts):
        # source: imutils package

        # if the length the contours tuple returned by cv2.findContours
        # is '2' then we are using either OpenCV v2.4, v4-beta, or
        # v4-official
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        # otherwise OpenCV has changed their cv2.findContours return
        # signature yet again and I have no idea WTH is going on
        else:
            raise Exception(
                (
                    "Contours tuple must have length 2 or 3, "
                    "otherwise OpenCV changed their cv2.findContours return "
                    "signature yet again. Refer to OpenCV's documentation "
                    "in that case"
                )
            )

        # return the actual contours array
        return cnts

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
        print("Quadrilateral is not a rectangle.")
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


class CropPage():
    def __init__(self, *args, **kwargs):
        self.morph_kernel = (5,5)

    def apply_filter(self, image):
        print("Entered CropPage preprocessor")
        image_mine = image.copy()
        image = normalize(cv2.GaussianBlur(image, (3, 3), 0))

        sheet = self.find_page(image)
        if len(sheet) == 0:
            print(
                f"\tError: Paper boundary not found for the given image ! \nHave you accidentally included CropPage preprocessor?"
            )
            return None

        print(f"Found page corners: \t {sheet.tolist()}")

        image = four_point_transform(image, sheet)
        image_mine = four_point_transform(image_mine, sheet)

        return image_mine

    def find_page(self, image):

        image = normalize(image)

        _ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_TRUNC)
        image = normalize(image)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)

        # Close the small holes, i.e. Complete the edges on canny image
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        edge = cv2.Canny(closed, 55, 185) # 185, 55 before

        # findContours returns outer boundaries in CW and inner ones, ACW.
        cnts = grab_contours(
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
        print("Dimensions are : ", sheet.shape)
        return sheet

def main(input_path):
    """Handles argument parsing, image loading, processing, and saving, with default path fallback."""
    # 1. Setup Argument Parser
    # parser = argparse.ArgumentParser(description="Automatically crops an image to the page boundary.")
    
    # # We now make both arguments optional by providing a default of None
    # parser.add_argument("input_path", type=str, nargs='?', default=None, help="Path to the input image file (e.g., .jpeg, .png).")
    # parser.add_argument("output_dir", type=str, nargs='?', default=None, help="Directory to save the cropped output image.")
    
    # args = parser.parse_args()
    
    # 2. Determine Paths (Use default if arguments are missing)
    # if args.input_path is None or args.output_dir is None:
    #     print("Arguments not provided. Falling back to default paths defined in the script.")
    #     input_path = DEFAULT_INPUT_PATH
    #     output_dir = DEFAULT_OUTPUT_DIR
    # else:
    #     input_path = args.input_path
    #     output_dir = args.output_dir
    

    # 3. Validate Paths
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please update DEFAULT_INPUT_PATH or provide command-line arguments.")
        return

    # if not os.path.isdir(output_dir):
    #     # Create output directory if it doesn't exist
    #     try:
    #         os.makedirs(output_dir)
    #         print(f"Created output directory: {output_dir}")
    #     except OSError as e:
    #         print(f"Error creating output directory: {e}")
    #         return

    # 4. Determine Output Filename
    # filename = os.path.basename(input_path)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    filename = "cropped_page_pic.jpeg"
    # base, ext = os.path.splitext(filename)
    # output_filename = f"{base}_cropped{ext}" # e.g., t1_p1_cropped.jpeg
    output_path = os.path.join(SCRIPT_DIR, filename)


    # 5. Image Processing
    cp = CropPage()
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return
        
    cropped_img = cp.apply_filter(img)
    cv2.imshow("Image of OMR before preprocessing", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("Image of OMR after preprocessing", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6. Save Output
    if cropped_img is not None:
        cv2.imwrite(output_path, cropped_img)
        print("-" * 50)
        print(f"Cropped image saved to: {output_path}")
        print(f"Output shape: {cropped_img.shape}")
    else:
        print("-" * 50)
        print("❌ Failure: Cropping filter returned no image.")

    return output_path

# --- Main Execution Block ---
# if __name__ == "__main__":
#     main()