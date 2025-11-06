import os
import json
from PIL import Image
from pdf2image import convert_from_path
import argparse # <-- NEW: Import argparse

# --- DEFAULT PATHS FOR NO-ARGUMENT RUNS ---
# !!! IMPORTANT: Update these paths to valid locations on your system for default operation !!!
DEFAULT_JSON = r"C:\Users\Navaneeth\BTP_1\Final_repo\Template_maker\Templates\json\t1.json"
DEFAULT_PDF = r"C:\Users\Navaneeth\BTP_1\Final_repo\Template_maker\Templates\pdf\t1.pdf"
DEFAULT_QR_FOLDER = r"C:\Users\Navaneeth\BTP_1\Final_repo\QR_code_pics"
DEFAULT_OUTPUT_FOLDER = r"C:\Users\Navaneeth\BTP_1\Final_repo\output_pdfs"
# ------------------------------------------

DPI = 300
A4_WIDTH_INCH = 8.27
A4_HEIGHT_INCH = 11.69
A4_WIDTH_PX = int(A4_WIDTH_INCH * DPI)
A4_HEIGHT_PX = int(A4_HEIGHT_INCH * DPI)

def generate_qr_pdfs(template_json_path, template_pdf_path, qr_folder, output_folder):
    """
    Inserts each QR image from qr_folder into the position of the QR placeholder in template_json,
    overlays it onto template.pdf, and saves each result in output_folder.
    """
    print("--- Starting PDF Generation ---")
    
    # 1. --- Load template JSON ---
    if not os.path.exists(template_json_path):
        raise FileNotFoundError(f"❌ Template JSON not found at: {template_json_path}")
        
    with open(template_json_path, 'r') as f:
        template_data = json.load(f)

    # 2. --- Find QR placeholder position ---
    qr_element = None
    
    template_data = template_data["images"] if "images" in template_data else template_data

    for el in template_data:
        # Assumes the QR placeholder image is named "QR.png" as per original code
        if el.get("file") == "QR.png":
            qr_element = el
            break

    if qr_element is None:
        raise ValueError("❌ No element with file='QR.png' found in template JSON.")

    # Apply slight padding/offset as in the original code
    qr_x = qr_element["x"] + 3
    qr_y = qr_element["y"] + 3
    qr_w = qr_element["width"] - 5
    qr_h = qr_element["height"] - 5
    
    print(f"✅ QR position found at real (x, y, w, h): ({qr_x}, {qr_y}, {qr_w}, {qr_h})")

    # 3. --- Ensure output folder exists ---
    os.makedirs(output_folder, exist_ok=True)
    print(f"📂 Output will be saved to: {output_folder}")

    # 4. --- Load base template (PDF background) ---
    if not os.path.exists(template_pdf_path):
        raise FileNotFoundError(f"❌ Template PDF not found at: {template_pdf_path}")
        
    try:
        pages = convert_from_path(template_pdf_path, dpi=300)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to convert PDF using pdf2image. Is the PDF valid and poppler installed? Error: {e}")
        
    base_template = pages[0].convert("RGB")
    # Resize ensures the template matches the expected A4_WIDTH_PX/A4_HEIGHT_PX dimensions
    base_template = base_template.resize((A4_WIDTH_PX, A4_HEIGHT_PX))
    print("✅ Base PDF template loaded and sized correctly.")

    # 5. --- Gather all QR images ---
    if not os.path.exists(qr_folder):
        raise FileNotFoundError(f"❌ QR image folder not found at: {qr_folder}")
        
    qr_images = [os.path.join(qr_folder, f) for f in os.listdir(qr_folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    qr_images.sort()

    if not qr_images:
        raise ValueError(f"❌ No QR images found in the provided qr_folder: {qr_folder}.")
    
    print(f"Found {len(qr_images)} QR images to process.")

    # 6. --- Generate a PDF for each QR ---
    for idx, qr_path in enumerate(qr_images, start=1):
        # print(f"🖼️ Inserting QR: {qr_path}")
        output_img = base_template.copy()

        # Open and resize QR
        qr_img = Image.open(qr_path).convert("RGBA")
        qr_img = qr_img.resize((qr_w, qr_h))

        # Paste QR into location using the QR's alpha channel (for transparency)
        output_img.paste(qr_img, (qr_x, qr_y), qr_img)

        # Save as PDF
        output_filename = os.path.join(output_folder, f"template_with_qr_{idx}.pdf")
        output_img.save(output_filename, "PDF", resolution=DPI)

    print(f"\n✅ Done! {len(qr_images)} PDFs created in '{output_folder}'.")


def main():
    """Handles argument parsing and calls the generator function."""
    parser = argparse.ArgumentParser(
        description="Generates a batch of PDFs by overlaying unique QR codes onto a base PDF template.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Define arguments with defaults
    parser.add_argument(
        "template_json_path", 
        type=str, 
        nargs='?', # Optional argument
        default=DEFAULT_JSON,
        help="Path to the template metadata JSON file (must contain 'QR.png').\n(Default: " + DEFAULT_JSON + ")"
    )
    parser.add_argument(
        "template_pdf_path", 
        type=str, 
        nargs='?', # Optional argument
        default=DEFAULT_PDF,
        help="Path to the base template PDF file.\n(Default: " + DEFAULT_PDF + ")"
    )
    parser.add_argument(
        "qr_folder", 
        type=str, 
        nargs='?', # Optional argument
        default=DEFAULT_QR_FOLDER,
        help="Path to the folder containing unique QR code images (.png, .jpg).\n(Default: " + DEFAULT_QR_FOLDER + ")"
    )
    parser.add_argument(
        "output_folder", 
        type=str, 
        nargs='?', # Optional argument
        default=DEFAULT_OUTPUT_FOLDER,
        help="Path to the directory where the output PDFs will be saved.\n(Default: " + DEFAULT_OUTPUT_FOLDER + ")"
    )

    args = parser.parse_args()
    
    try:
        generate_qr_pdfs(
            template_json_path=args.template_json_path,
            template_pdf_path=args.template_pdf_path,
            qr_folder=args.qr_folder,
            output_folder=args.output_folder
        )
    except Exception as e:
        print(f"\nAn error occurred during batch generation: {e}")
        print("Please check file paths, folder contents, and ensure 'poppler' is installed for pdf2image.")

if __name__ == "__main__":
    main()
