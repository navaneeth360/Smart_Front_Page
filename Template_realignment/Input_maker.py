import tkinter as tk
from tkinter import filedialog, messagebox
import os
import Cropper
import template_realignment as tr
import json


config_data = {
                "dimensions": {
                    "display_height": 1600,
                    "display_width": 1200,
                    "processing_height": 1600,
                    "processing_width": 1200
                },
                "outputs": {
                    "show_image_level": 5
                }
            }


def select_input_file():
    file_path = filedialog.askopenfilename(
        title="Select Image of front page",
        filetypes=[("JPEG Images", "*.jpg;*.jpeg")]
    )
    if not file_path:
        messagebox.showwarning("Warning", "No file selected!")
        return None
    messagebox.showinfo("Selected File", f"Input file:\n{file_path}")
    return file_path

def select_image_folder(given_path):
    folder_path = filedialog.askdirectory(
        initialdir= os.path.dirname(given_path),
        title="Select Image Directory"
    )
    if not folder_path:
        messagebox.showwarning("Warning", "No image folder selected!")
        return None
    messagebox.showinfo("Input Folder", f"Input folder:\n{folder_path}")
    return folder_path


def run_workflow():
    # First accept the input image path
    input_file = select_input_file()
    if not input_file:
        return
    
    # Then run crop page
    cropped_img_path = Cropper.main(input_file)

    # messagebox.showinfo("Processing", "Processing input file...")
    root.destroy()


    # Realign template, save it and return the template
    saved_template = tr.init_gui(cropped_img_path)

    root.mainloop()

    output_dir = select_image_folder(input_file)

    t_file = "template.json"
    output_filepath = os.path.join(output_dir, t_file)

    try:
        with open(output_filepath, 'w') as json_file:
            json.dump(saved_template, json_file, indent=4)
    except IOError as e:
        print(f"An error occurred while writing the file: {e}")

    c_file = "config.json"
    output_filepath = os.path.join(output_dir, c_file)

    try:
        with open(output_filepath, 'w') as json_file:
            json.dump(config_data, json_file, indent=4)
    except IOError as e:
        print(f"An error occurred while writing the file: {e}")
    
    print("Successfully created input folder : ", output_dir)
    print("Use this as inputDir for main.py in OMR_checker :)")

    try:
        os.remove(cropped_img_path)
        # print(f"Successfully deleted: {cropped_img_path}")
    except OSError as e:
        print(f"Error deleting file '{cropped_img_path}': {e}")



# Create and run the GUI main window
root = tk.Tk()
root.title("Input Directory Creation")
root.geometry("300x150")

btn = tk.Button(root, text="Start Realignment and Directory Creation", command=run_workflow, height=2)
btn.pack(expand=True)

root.mainloop()
