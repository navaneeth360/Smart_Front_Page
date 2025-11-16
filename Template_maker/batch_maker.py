import os
import json
from PIL import Image
from pdf2image import convert_from_path
import argparse 
from PIL import ImageDraw, ImageFont

# --- DEFAULT PATHS FOR NO-ARGUMENT RUNS ---
# !!! IMPORTANT: Update these paths to valid locations on your system for default operation !!!
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "Templates\\json\\fin1.json") # r"C:\Users\Navaneeth\BTP_1\Final_repo\Template_maker\Templates\json\t1.json"
DEFAULT_PDF = os.path.join(SCRIPT_DIR, "Templates\\pdf\\fin1.pdf")  #r"C:\Users\Navaneeth\BTP_1\Final_repo\Template_maker\Templates\pdf\t1.pdf"

DEFAULT_QR_FOLDER = r"C:\Users\Navaneeth\BTP_1\Final_repo\QR_code_pics"
DEFAULT_OUTPUT_FOLDER = r"C:\Users\Navaneeth\BTP_1\Final_repo\output_pdfs2"
DEFAULT_IMAGE_DIR   = os.path.join(SCRIPT_DIR, "Field_images")
# ------------------------------------------

DPI = 300
A4_WIDTH_INCH = 8.27
A4_HEIGHT_INCH = 11.69
A4_WIDTH_PX = int(A4_WIDTH_INCH * DPI)
A4_HEIGHT_PX = int(A4_HEIGHT_INCH * DPI)

def create_pdf_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    texts  = data.get("texts", [])

    # Create blank A4 at 300 DPI
    output = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), "white")
    draw = ImageDraw.Draw(output)

    # -------------------------------------------
    # Draw images exactly as pick_and_drop export
    # -------------------------------------------
    for item in images:
        if item["file"] == "QR.png":
            continue

        # Try JSON-folder first
        json_dir = os.path.dirname(json_path)
        candidate1 = os.path.join(json_dir, item["file"])
        candidate2 = os.path.join(DEFAULT_IMAGE_DIR, item["file"])

        if os.path.exists(candidate1):
            img_path = candidate1
        elif os.path.exists(candidate2):
            img_path = candidate2
        else:
            raise FileNotFoundError(
                f"Image '{item['file']}' not found in:\n{candidate1}\n{candidate2}"
            )

        img = Image.open(img_path).convert("RGB")
        img = img.resize((item["width"], item["height"]))
        output.paste(img, (item["x"], item["y"]))

        # ---- Draw bounding box ----
        draw.rectangle(
            [
                (item["x"], item["y"]),
                (item["x"] + item["width"], item["y"] + item["height"])
            ],
            outline="black",
            width=2
        )

    # -------------------------------------------
    # Draw text exactly like pick_and_drop
    # -------------------------------------------
    for t in texts:
        # Extract font name & size
        try:
            parts = t["font"].split()
            font_name = parts[0]
            font_size_display = int(parts[1])
        except:
            font_name = "Arial"
            font_size_display = 12

        # pick_and_drop converts UI font sizes to PDF sizes
        font_size = int(font_size_display / 0.22)

        try:
            font = ImageFont.truetype(f"{font_name.lower()}.ttf", font_size)
        except:
            font = ImageFont.load_default()

        draw.text((t["x"], t["y"]), t["content"], fill="black", font=font)


    # Save PDF
    outfile = os.path.join(SCRIPT_DIR, "_temp_json_exact.pdf")
    output.save(outfile, "PDF", resolution=DPI)
    return outfile




def generate_qr_pdfs(template_json_path, template_pdf_path, qr_folder, output_folder, use_pdf1):
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
    if use_pdf1 == "True":
        # Use PDF directly
        if not os.path.exists(template_pdf_path):
            raise FileNotFoundError(f"❌ Template PDF not found: {template_pdf_path}")

        try:
            pages = convert_from_path(template_pdf_path, dpi=300)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to convert PDF using pdf2image. Is the PDF valid and poppler installed? Error: {e}")
        base_template = pages[0].convert("RGB")
        base_template = base_template.resize((A4_WIDTH_PX, A4_HEIGHT_PX))

    else:
        # Build PDF from JSON metadata
        print("⚙️  Building PDF from JSON layout (use_pdf=False)")
        temp_pdf = create_pdf_from_json(template_json_path)
        try:
            pages = convert_from_path(temp_pdf, dpi=300)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to convert PDF using pdf2image. Is the PDF valid and poppler installed? Error: {e}")
        base_template = pages[0].convert("RGB")
        base_template = base_template.resize((A4_WIDTH_PX, A4_HEIGHT_PX))

    # if not os.path.exists(template_pdf_path):
    #     raise FileNotFoundError(f"❌ Template PDF not found at: {template_pdf_path}")
        
    try:
        pages = convert_from_path(template_pdf_path, dpi=300)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to convert PDF using pdf2image. Is the PDF valid and poppler installed? Error: {e}")
        
    # base_template = pages[0].convert("RGB")
    # # Resize ensures the template matches the expected A4_WIDTH_PX/A4_HEIGHT_PX dimensions
    # base_template = base_template.resize((A4_WIDTH_PX, A4_HEIGHT_PX))
    # print("✅ Base PDF template loaded and sized correctly.")

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
        if use_pdf1 == "False":
            draw = ImageDraw.Draw(output_img)
            draw.rectangle(
                [
                    (qr_x, qr_y),
                    (qr_x + qr_img.width, qr_y + qr_img.height)
                ],
                outline="black",
                width=2
            )

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
        "--template_json_path", 
        type=str, 
        nargs='?', # Optional argument
        default=DEFAULT_JSON,
        help="Path to the template metadata JSON file (must contain 'QR.png').\n(Default: " + DEFAULT_JSON + ")"
    )
    parser.add_argument(
        "--template_pdf_path", 
        type=str, 
        nargs='?', # Optional argument
        default=DEFAULT_PDF,
        help="Path to the base template PDF file.\n(Default: " + DEFAULT_PDF + ")"
    )
    parser.add_argument(
        "--qr_folder", 
        type=str, 
        nargs='?', # Optional argument
        default=DEFAULT_QR_FOLDER,
        help="Path to the folder containing unique QR code images (.png, .jpg).\n(Default: " + DEFAULT_QR_FOLDER + ")"
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        nargs='?', # Optional argument
        default=DEFAULT_OUTPUT_FOLDER,
        help="Path to the directory where the output PDFs will be saved.\n(Default: " + DEFAULT_OUTPUT_FOLDER + ")"
    )
    parser.add_argument(
        "--use_pdf",
        nargs='?', # Optional argument
        default="True",
        help="If set, use the PDF directly. If NOT set, PDF is built from the JSON."
    )


    args = parser.parse_args()
    
    try:
        generate_qr_pdfs(
            template_json_path=args.template_json_path,
            template_pdf_path=args.template_pdf_path,
            qr_folder=args.qr_folder,
            output_folder=args.output_folder,
            use_pdf1 = args.use_pdf
        )
        if args.use_pdf == "False":
            try:
                pdf_file_path = os.path.join(SCRIPT_DIR, "_temp_json_exact.pdf")
                os.remove(pdf_file_path)
                print(f"Successfully deleted: {pdf_file_path}")
            except OSError as e:
                print(f"Error deleting file '{pdf_file_path}': {e}")

    except Exception as e:
        print(f"\nAn error occurred during batch generation: {e}")
        print("Please check file paths, folder contents, and ensure 'poppler' is installed for pdf2image.")

if __name__ == "__main__":
    main()
