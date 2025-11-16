
# 📄 Smart Front Page: Automated Academic Evaluation System

The Smart Front Page is a comprehensive, all-in-one system designed to automate the manual and error-prone process of handling front page details, OMR responses, and marks for large-scale academic examinations. Built upon an existing OMR checking foundation, this project introduces custom GUIs, QR code integration, and advanced recognition capabilities for text and digits.

-----

## 🚀 Key Features

The system is composed of four interconnected modules:

  * **Template Maker GUI:** An interactive, user-friendly interface built with Tkinter to design custom A4-sized templates using a **pick-and-place mechanism** for text, OMR bubbles, and images (like QR codes).
  * **QR Generator:** Creates a batch of unique QR code images for each student, embedding their **roll number** (the primary key) for reliable identification.
  * **Template Realigner GUI:** A fine-tuning tool that overlays the template onto a scanned image, allowing for precise drag-and-resize adjustments to field boundaries, significantly improving extraction accuracy.
  * **Expanded Evaluation:** Extends the core OMR functionality to accurately process three additional field types from the scanned images:
      * **Text Recognition:** Processes text fields using a pre-trained **EMNIST neural network model** as well as an OCR.
      * **Digit Recognition:** Achieves high accuracy ($\sim$99%) for reading digits using an **MNIST-based neural network model**.
      * **QR Code Recognition:** Instantly processes the generated QR codes once scanned using the $\texttt{qrcode}$ library in Python.

-----
The Template\_maker folder contains the Template Maker GUI and QR Generator modules, OMR_checker has the Evaluation module codes and Template\_realignment has the codes for the Template Realigner GUI.

## 🛠️ Installation and Setup

### Prerequisites

You need **Python 3.6+** and the dependencies listed in $\texttt{requirements.txt}$.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/navaneeth360/Smart_Front_Page.git
    cd Smart_Front_Page
    ```

2.  **Install Dependencies:**
    All required packages are installed using pip.

    ```bash
    pip install -r requirements.txt
    ```

### Configuration Notes

Before running the workflow, ensure you adjust the file paths and column names in the following files as per the usage instructions:

  * **qr\_generator.py:** Provide the path to the CSV file, the output directory, and the name of the roll number column at the top of the code before running it.
  * **pick\_and\_drop.py:** The default directories at the top can be changed, but this is not recommended.
  * **batch\_maker.py:** File paths can be modified at the top or passed as command-line arguments.

-----

## ⚙️ Usage Workflow

The system is designed to be used in a few easy, sequential steps to produce the final evaluation data.

### Step 1: Design Template

Run the interactive Template Maker GUI.

```bash
python pick_and_drop.py
```

  * **Action:** Insert fields, images, and text onto the A4 template.
  * **Output:** A required **JSON metadata file** (for future processing) and an optional PDF preview.

### Step 2: Generate Student Data

Generate the unique QR code images needed for the batch printing.

```bash
python qr_generator.py
```

  * **Input:** Student data CSV file path.
  * **Output:** A set of unique QR code image files, one for each student, with their roll number printed below.

### Step 3: Assemble Front Page PDFs

Combine the template metadata and the unique QR codes to create the final, printable front pages.

```bash
# Example: Using the JSON metadata (recommended)
python batch_maker.py path/to/template.json path/to/qr/folder/
```

  * **Action:** This generates a final batch of front page PDFs. These sheets are then printed and attached to the answer booklets before the exam.

### Step 4: Capture and Realign Template

After the exam and evaluation, the evaluator takes a picture of the sheet. Then, run the Input Maker to process and align a sample image.

```bash
# Ensure you are in the Template_realignment folder
python Input_maker.py 
```

  * **Action:** The script prompts for a sample image and the template JSON. It automatically calls $\texttt{Cropper.py}$ to crop the page. The cropped image is passed to the template\_realignment.py GUI.
  * **Realignment:** Drag and resize the boxes in the GUI to perfectly match the fields in the scanned image.
  * **Output:** An **updated JSON template** is saved, ready for the evaluation module.

### Step 5: Run Evaluation

The input directory containing the scanned images and the finalized JSON files is ready for the evaluation software.

  * **Action:** The software goes through each image, extracts the data from the various fields (text, OMR, QR), and saves the data.
  * **Output:** A single CSV file containing the final results.

-----

## 🔍 Core Image Processing ($\texttt{Cropper.py}$)

The document cropping step is critical for accurate evaluation. It uses a robust computer vision pipeline:

1.  **Smoothing and Normalization:** Applies Gaussian Blur to smoothen the image and normalization to improve edge detection.
2.  **Feature Isolation:** Thresholding is done to remove very bright pixels, followed by **Morphological Closing** (dilation then erosion) to close small gaps in the document boundary.
3.  **Edge Detection:** **Canny Edge Detection** is performed, which works by calculating the gradient at every pixel, followed by Non-Maximum Suppression (NMS) and Hysteresis Thresholding.
4.  **Contour Analysis:** A contour analysis is done to identify closed curves, and the largest curves are selected and approximated to a polygon using a **convex hull**.
5.  **Perspective Correction:** Once a suitable rectangle has been identified, a **four-point transform** is performed to deskew and crop the image into the final rectangular output.

-----

## 🔮 Future Enhancements

  * **Moodle Integration:** Automate mark uploads using the final output CSV file.
  * **Improved Accuracy:** Improve the accuracy of text identification by trying different neural network architectures.
  * **Mobile Application:** Design a mobile application to facilitate the upload of the scanned front page images.
  * **Custom Field GUI:** Create a GUI to make custom fields with ease.
