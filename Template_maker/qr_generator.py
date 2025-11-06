import qrcode
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
import argparse # <-- NEW: Import argparse

# --- DEFAULT PATHS FOR NO-ARGUMENT RUNS ---
# !!! IMPORTANT: Update these paths for quick testing !!!
DEFAULT_CSV_PATH = r"C:\Users\Navaneeth\BTP_1\QR_scanner\Database.csv"
DEFAULT_OUTPUT_DIR = r"C:\Users\Navaneeth\BTP_1\Final_repo\QR_code_pics"
DEFAULT_ROLL_KEY = "roll"
# ------------------------------------------

def generate_qrs(path_to_csv, output_dir, roll_key, add_roll_no=True): 
    """
    Generates unique QR codes based on data in a CSV file and saves them.
    
    Args:
        path_to_csv (str): Path to the input CSV file.
        output_dir (str): Directory where QR code images will be saved.
        add_roll_no (bool): Whether to overlay the roll number text onto the QR image.
    """
    try:
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' successfully created.")
    except FileExistsError: # Catch the specific exception
        print(f"Output directory '{output_dir}' already exists.")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    try:
        df = pd.read_csv(path_to_csv)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {path_to_csv}")
        return
        
    print(df.head())

    # Find which column is the roll number column
    df.columns = df.columns.str.lower()
    key = roll_key
    mask = df.columns.str.contains(key)
    print("mask : ", mask)
    roll_ind = -1 # Initialize with invalid index
    for i in range(len(mask)):
        if (mask[i] == True):
            roll_ind = i
            break
    
    if roll_ind == -1:
        print("Error: Could not find a column containing 'roll'. Exiting.")
        return
        
    roll_name = df.columns[roll_ind]
    df[roll_name] = df[roll_name].astype(str).str.upper() # Ensure roll number is treated as string

    for index, row in df.iterrows(): 
        if index>10: 
            break
            
        data = list(row.values)
        # print(data, type(data))
        # print(data[0])
        
        filename = os.path.join(output_dir, f"qr_{data[roll_ind]}.png")
        
        # generate qr code
        # The entire row of data is used for QR code generation, as per original logic: qrcode.make(data)
        img = qrcode.make(data)

        if (add_roll_no):
            # Roll number to overlay
            text = data[roll_ind]
            
            img = img.convert("RGB") 
            try:
                # Try to load a common font
                font = ImageFont.truetype("arial.ttf", 70)
            except IOError:
                # Fallback font
                font = ImageFont.load_default()

            draw = ImageDraw.Draw(img)
            # Use getbbox for PIL versions > 9.1.0, otherwise the original logic might be needed,
            # but bbox is calculated relative to (0,0) in either case.
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
            except AttributeError:
                # Fallback for older PIL versions (less accurate)
                text_width, text_height = draw.textsize(text, font=font)
                bbox = (0, 0, text_width, text_height)
                
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            qr_width, qr_height = img.size
            # Add padding of 20 below the QR code for the text
            new_height = qr_height + text_height + 20

            # Create new image and paste QR
            new_img = Image.new("RGB", (qr_width, new_height), "white")
            new_img.paste(img, (0, 0))

            # Draw text
            draw_new = ImageDraw.Draw(new_img)
            text_x = (qr_width - text_width) // 2
            # Position text 10 pixels below the original QR code image area
            text_y = qr_height - 30 
            draw_new.text((text_x, text_y), text, fill="black", font=font)
            img = new_img

        img.save(filename)
        print(f"Generated {filename}")
    
    print("\n✅ QR Code generation complete.")

def main():
    """Handles argument parsing and calls the generation function."""
    parser = argparse.ArgumentParser(
        description="Generates QR codes from a CSV file, using the 'roll' column to name files and optionally overlay text."
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        "csv_path", 
        type=str, 
        nargs='?', # Makes it optional
        default=DEFAULT_CSV_PATH,
        help=f"Path to the input CSV file. (Default: {DEFAULT_CSV_PATH})"
    )
    parser.add_argument(
        "output_dir", 
        type=str, 
        nargs='?', # Makes it optional
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the generated QR code images. (Default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "roll_key", 
        type=str, 
        nargs='?', 
        default=DEFAULT_ROLL_KEY,
        help=f"The key string used to identify the roll number column in the CSV (e.g., 'roll', 'id'). (Default: {DEFAULT_ROLL_KEY})"
    )
    parser.add_argument(
        "--no-roll-no", 
        action='store_false', 
        dest='add_roll_no',
        help="If set, prevents the overlaying of roll number text onto the QR code image."
    )

    args = parser.parse_args()
    
    generate_qrs(
        path_to_csv=args.csv_path,
        output_dir=args.output_dir,
        roll_key=args.roll_key,
        add_roll_no=args.add_roll_no
    )

if __name__ == "__main__":
    main()