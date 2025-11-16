import fitz  # PyMuPDF
import cv2
import numpy as np

def draw_rectangles_on_pdf(pdf_path, rectangles, target_width=800, render_scale=2.0):
    """
    Reads the first page of a PDF, resizes it to a target width, 
    draws specified rectangles, and displays the result.

    Args:
        pdf_path (str): The file path to the input PDF.
        rectangles (list): A list of dictionaries, where each dict specifies a
                           rectangle to draw: [{'x': 100, 'y': 50, 'w': 200, 'h': 150, 'color': (0, 0, 255)}]
        target_width (int): The desired width of the final image (height is calculated to maintain aspect ratio).
        render_scale (float): Initial scaling factor for rendering the PDF (ensures good quality).
    """
    try:
        # 1. Open the PDF and get the first page
        doc = fitz.open(pdf_path)
        page = doc[0]

        # 2. Render the page to an initial high-resolution pixmap
        matrix = fitz.Matrix(render_scale, render_scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        
        # Convert to BGR for OpenCV
        image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 3. Resize the image while maintaining aspect ratio
        original_width = image.shape[1]
        original_height = image.shape[0]
        
        # Calculate new height
        if original_width > 0:
            ratio = target_width / original_width
            target_height = 3507
            # target_height = int(original_height * ratio)
        else:
            print("Error: Original image width is zero. Cannot resize.")
            return

        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        print(f"Image resized from ({original_width}x{original_height}) to ({target_width}x{target_height}).")

        # 4. Calculate the total scaling factor for coordinate mapping
        # Total_Scale = (Render_Scale) * (Resize_Scale)
        total_x_scale = target_width / page.rect.width
        total_y_scale = target_height / page.rect.height

        # 5. Draw the rectangles on the resized image
        for rect in rectangles:
            # Scale and convert floating point coordinates to integers
            x = int(rect['x']) # * total_x_scale)
            y = int(rect['y']) # * total_y_scale)
            w = int(rect['w']) # * total_x_scale)
            h = int(rect['h']) # * total_y_scale)
            color = rect.get('color', (0, 255, 0)) # Default to green (BGR)
            thickness = rect.get('thickness', 3) # Default thickness

            # Define the top-left and bottom-right corners for cv2.rectangle()
            top_left = (x, y)
            bottom_right = (x + w, y + h)

            cv2.rectangle(
                img=resized_image, 
                pt1=top_left, 
                pt2=bottom_right, 
                color=color, 
                thickness=thickness
            )
            print(f"Drawing rectangle based on PDF coordinates ({rect['x']:.2f}, {rect['y']:.2f})")

        # 6. Display the image
        display_width = int(target_width * display_scale_factor)
        display_height = int(target_height * display_scale_factor)
        
        if display_scale_factor != 1.0:
            final_display_image = cv2.resize(resized_image, (display_width, display_height), interpolation=cv2.INTER_AREA)
            print(f"Display window scaled down to: ({display_width}x{display_height}).")
        else:
            final_display_image = resized_image
            
        # 7. Display the image
        cv2.imshow("PDF Page with Rectangles (Scaled Display)", final_display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    display_scale_factor = 0.25  # Scale down for display purposes
    PDF_FILE = r"C:\Users\Navaneeth\BTP_1\Template_maker\OMR_options\testing_12.pdf"
    
    # Define the rectangles using points relative to the original PDF page size.
    # RECTANGLES_TO_DRAW = [
    #     {'x': 366, 'y': 555, 'w': 706, 'h': 750, 'color': (255, 0, 0)},                   # Blue box
    #     {'x': 410, 'y': 600, 'w': 70, 'h': 70, 'color': (255, 0, 0)},                  # TextBox
    #     {'x': 1546, 'y': 963, 'w': 433, 'h': 460, 'color': (0, 255, 0), 'thickness': 2},   # Green box
    #     {'x': 366, 'y': 555, 'w': 706, 'h': 750, 'color': (255, 0, 0)},
    #     {'x': 135, 'y': 223, 'w': 536, 'h': 613, 'color': (255, 0, 0)}
    # ]
    
    RECTANGLES_TO_DRAW = [
        {'x': 151, 'y': 111, 'w': 610, 'h': 703, 'color': (255, 0, 0)},                   # Blue box
        {'x': 815, 'y': 123, 'w': 376, 'h': 446, 'color': (0, 255, 0)}
    ]

    xi = 841
    yi = 139
    delx = 54
    dely = 66

    for i in range(6):
        d = {'x': xi + i*delx, 'y': yi, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
        RECTANGLES_TO_DRAW.append(d)

    xi = 194
    yi = 137
    delx = 88
    dely = 105

    for i in range(6):
        d = {'x': xi + i*delx, 'y': yi, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
        RECTANGLES_TO_DRAW.append(d)

    xi = 384
    yi = 265
    delx = 60
    dely = 37
    gapy = 17
    gapx = 88

    for j in range(4):
        for i in range(10):
            d = {'x': xi + j*gapx, 'y': yi + i*gapy + i*dely, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
            RECTANGLES_TO_DRAW.append(d)
    
    xi = 143 + 815
    yi = 98 + 123
    delx = 37
    dely = 23
    gapy = 11
    gapx = 54

    for j in range(4):
        for i in range(10):
            d = {'x': xi + j*gapx, 'y': yi + i*gapy + i*dely, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
            RECTANGLES_TO_DRAW.append(d)

    # NAME
    xi = 141 + 30
    yi = 865 + 21
    delx = 68
    dely = 100

    for i in range(25):
        d = {'x': xi + i*delx, 'y': yi, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
        RECTANGLES_TO_DRAW.append(d)

    xi = 145 + 26
    yi = 1061 + 18
    delx = 59.5
    dely = 86

    for i in range(25):
        d = {'x': xi + i*delx, 'y': yi, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
        RECTANGLES_TO_DRAW.append(d)
    
    # ROLL NUMBER
    xi = 1282 + 34
    yi = 115 + 22
    delx = 65
    dely = 84

    for i in range(7):
        d = {'x': xi + i*delx, 'y': yi, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
        RECTANGLES_TO_DRAW.append(d)

    xi = 1863 + 18
    yi = 116 + 13
    delx = 46
    dely = 60

    for i in range(7):
        d = {'x': xi + i*delx, 'y': yi, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
        RECTANGLES_TO_DRAW.append(d)

    xi = 1282 + 41
    yi = 115 + 128
    delx = 42
    dely = 30
    gapy = 14
    gapx = 64

    for j in range(7):
        for i in range(10):
            d = {'x': xi + j*gapx, 'y': yi + i*gapy + i*dely, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
            RECTANGLES_TO_DRAW.append(d)
    
    xi = 1863 + 29
    yi = 116 + 92
    delx = 30
    dely = 21.5
    gapy = 10
    gapx = 46

    for j in range(7):
        for i in range(10):
            d = {'x': xi + j*gapx, 'y': yi + i*gapy + i*dely, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
            RECTANGLES_TO_DRAW.append(d)
    
    # Marks
    xi = 185 + 43
    yi = 1297 + 24
    delx = 92
    dely = 110

    for i in range(6):
        d = {'x': xi + i*delx, 'y': yi, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
        RECTANGLES_TO_DRAW.append(d)
    
    xi = 913 + 32
    yi = 1295 + 16
    delx = 69
    dely = 74

    for i in range(6):
        d = {'x': xi + i*delx, 'y': yi, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
        RECTANGLES_TO_DRAW.append(d)

    xi = 185 + 61
    yi = 1297 + 160
    delx = 61
    dely = 38
    gapy = 17
    gapx = 92

    for j in range(6):
        for i in range(10):
            d = {'x': xi + j*gapx, 'y': yi + i*gapy + i*dely, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
            RECTANGLES_TO_DRAW.append(d)

    xi = 913 + 46
    yi = 1295 + 108
    delx = 46
    dely = 26
    gapy = 11
    gapx = 69

    for j in range(6):
        for i in range(10):
            d = {'x': xi + j*gapx, 'y': yi + i*gapy + i*dely, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
            RECTANGLES_TO_DRAW.append(d)

    # QR
    xi = 1458 + 35 - 50
    yi = 1293 + 35 - 50
    delx = 400 + 100
    dely = 370 + 100

    for i in range(1):
        d = {'x': xi + i*delx, 'y': yi, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
        RECTANGLES_TO_DRAW.append(d)

    xi = 1533 + 47 - 64
    yi = 1886 + 32  - 48
    delx = 538 + 128
    dely = 341 + 96

    for i in range(1):
        d = {'x': xi + i*delx, 'y': yi, 'w': delx, 'h': dely, 'color': (255, 0, 0)}
        RECTANGLES_TO_DRAW.append(d)

    
    
    # Call the function, resizing the final displayed image to 1024 pixels wide.
    draw_rectangles_on_pdf(PDF_FILE, RECTANGLES_TO_DRAW, target_width=2481)