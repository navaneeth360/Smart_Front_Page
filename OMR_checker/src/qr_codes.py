import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import qrcode
import os
from PIL import Image, ImageDraw, ImageFont
import src.constants as constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def get_qr_data(image):
    # Send the image of the qr cropped or uncropped
    detector = cv2.QRCodeDetector()
    data, bbox, straight_qrcode = detector.detectAndDecode(image)
    if bbox is not None:
        print(f"QRCode data:\n{data}")
    else:
        print("No data obtained from QR code")
    return data

def generate_qrs(path_to_csv = r"Database.csv", output_dir = "QR_code_pics", add_roll_no = True):
    try:
        os.makedirs(output_dir)
        print("Output directory successfully created")
    except:
        print("Output directory already exists")

    df = pd.read_csv(path_to_csv)
    print(df.head())

    # Find which column is the roll number column, search for roll and choose the first column that matches
    df.columns = df.columns.str.lower()
    key = "roll"
    mask = df.columns.str.contains(key)      # boolean array
    print("mask : ", mask)
    roll_ind = 1
    for i in range(len(mask)):
        if (mask[i] == True):
            roll_ind = i
            break
    # print("roll number column is : ", roll_ind)
    roll_name = df.columns[roll_ind]
    df[roll_name] = df[roll_name].str.upper()

    for index, row in df.iterrows():
        if index>10:
            break
        data = list(row.values)
        print(data, type(data))
        print(data[0])
        
        filename = f"{output_dir}/qr_{data[roll_ind]}.png"
        # generate qr code
        img = qrcode.make(data)

        if (add_roll_no):
            img = img.convert("RGB") 
            text = data[roll_ind]
            try:
                font = ImageFont.truetype("arial.ttf", 70)
            except:
                font = ImageFont.load_default()

            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            qr_width, qr_height = img.size
            new_height = qr_height + text_height + 20

            # Create new image and paste QR
            new_img = Image.new("RGB", (qr_width, new_height), "white")
            new_img.paste(img, (0, 0))

            # Draw text
            draw_new = ImageDraw.Draw(new_img)
            text_x = (qr_width - text_width) // 2
            text_y = qr_height - 30
            draw_new.text((text_x, text_y), text, fill="black", font=font)
            img = new_img

        img.save(filename)


def preprocess_digit_image(img):

    cv2.imshow("Image of character before", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Threshold and invert (make sure digit is white)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow("Image of character after invert", img_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Resize to 28x28
    img_resized = cv2.resize(img_bin, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imshow("Image of character after resize", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Normalize
    img_norm = img_resized.astype("float32") / 255.0

    # Add channel and batch dimension
    img_input = np.expand_dims(img_norm, axis=(0, -1))
    return img_input

import cv2
import numpy as np
import torch

def preprocess_character_image(img):
    """
    Preprocesses a grayscale handwritten character image to EMNIST style:
      - grayscale input
      - optional binary inversion (white bg → black bg)
      - resized to 28x28
      - rotated + flipped (to match EMNIST orientation)
      - normalized using EMNIST mean/std
      - returns a torch tensor of shape [1, 1, 28, 28]
    """

    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize to 28x28
    img_resized = cv2.resize(img_bin, (28, 28), interpolation=cv2.INTER_AREA)

    # Match EMNIST orientation: rotate 90° counterclockwise + flip horizontally
    # img_corrected = np.flip(np.rot90(img_resized, k=1), axis=1)

    # Convert to float32 and normalize (mean=0.1307, std=0.3081)
    img_norm = img_resized.astype(np.float32) / 255.0
    # img_norm = (img_norm - 0.1307) / 0.3081

    # Convert to torch tensor of shape [1, 1, 28, 28]
    tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)

    return tensor




# try:
#     # Use compile=False if you only need prediction and don't want to re-run compilation
#     cnn_model = load_model(constants.CNN_MODEL_LOAD_PATH, compile=False)
#     print(f"✅ Model successfully loaded from {constants.CNN_MODEL_LOAD_PATH}")
# except Exception as e:
#     print(f"❌ Error loading the model: {e}")
#     print("Please ensure you have run the training script to create 'emnist_cnn_model.h5'.")
#     cnn_model = None 

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 47)  # 0–9 + A–Z

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(-1, 64 * 5 * 5)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

try:
    # Use compile=False if you only need prediction and don't want to re-run compilation
    # device = ("cuda" if torch.cuda.is_available() else "cpu")
    # cnn_model = CNN().to(device)
    # cnn_model.load_state_dict(torch.load(constants.CNN_MODEL_LOAD_PATH))
    # cnn_model.eval()
    cnn_model = load_model(constants.CNN_MODEL_LOAD_PATH, compile=False)
    print(f"✅ Model successfully loaded from {constants.CNN_MODEL_LOAD_PATH}")
except Exception as e:
    print(f"❌ Error loading the model: {e}")
    print("Please ensure you have run the training script to create 'emnist_cnn_model.h5'.")
    cnn_model = None


try:
    # Use compile=False if you only need prediction and don't want to re-run compilation
    mnist_model = load_model(constants.MNIST_MODEL_LOAD_PATH, compile=False)
    print(f"✅ Model successfully loaded from {constants.MNIST_MODEL_LOAD_PATH}")
except Exception as e:
    print(f"❌ Error loading the model: {e}")
    print("Please ensure you have run the training script to create 'emnist_cnn_model.h5'.")
    mnist_model = None 


def recognize_number(img):
    img_input = preprocess_digit_image(img)
    pred = mnist_model.predict(img_input)
    print("Prediction is : ", pred)
    digit = np.argmax(pred)
    confidence = np.max(pred)
    print("Digit and confidence : ", digit, confidence)
    return digit


def recognize_character(img):
    # img = img.transpose(2, 3).flip(2)
    img = preprocess_digit_image(img)
    # with torch.no_grad():
    #     output = cnn_model(img)
    #     pred = output.argmax(dim=1).item()
    #     print("Predicted:", constants.CHAR_MAP[pred])
    pred = cnn_model.predict(img)
    pred = np.argmax(pred)
    print("Prediction is : ", pred, constants.CHAR_MAP[pred])
    return constants.CHAR_MAP[pred]


def recognize_character_tesseract(img):
    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional binarization for clarity
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("Image of character after invert", img_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Run Tesseract OCR (single character mode)
    config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = pytesseract.image_to_string(img_bin, config=config)

    print("OCR Result:", text)

    # Clean up output
    return text.strip()[0] if text else ""



# Function to read characters
# def recognize_character_old(img):
#     img = preprocess_for_emnist(img)

#     cv2.imshow("Image of character", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     if cnn_model is None:
#         return "ERROR: Model is not loaded."

#     # --- Preprocessing the input image ---
    
#     # 1. Ensure it's a NumPy array for consistent processing
#     img = np.array(img, dtype='float32')

#     # 2. Reshape to (28, 28, 1) if it's currently (28, 28)
#     if img.shape == (28, 28):
#         img = img.reshape(28, 28, 1)

#     # # 3. Add the batch dimension: (1, 28, 28, 1) 
#     # Keras models always expect a batch of images, even if it's just one.
#     img = np.expand_dims(img, axis=0)

#     # 4. Normalize the image if it's not already in the 0-1 range (0-255 is common raw data)
#     if np.max(img) > 1.0:
#         img /= 255.0

#     try:
#         # Get the prediction probabilities
#         predictions = cnn_model.predict(img, verbose=0)
        
#         # Get the index (class) with the highest probability
#         predicted_index = np.argmax(predictions[0])
#         print("Predicted index is ", predicted_index)
        
#         # Map the index back to the character
#         predicted_char = constants.EMNIST_MAPPING.get(predicted_index, 'Unknown')
        
#         return predicted_char
    
#     except Exception as e:
#         print("Prediction Error", {e})
#         return None