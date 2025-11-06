import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import os
import json
import scaling_factors as sf

# --- Constants must match pick_and_dropf.py ---
DPI = 300
A4_WIDTH_INCH = 8.27
A4_HEIGHT_INCH = 11.69
A4_WIDTH_PX = int(A4_WIDTH_INCH * DPI)
A4_HEIGHT_PX = int(A4_HEIGHT_INCH * DPI)

DISPLAY_SCALE = 0.3
DISPLAY_W = int(A4_WIDTH_PX * DISPLAY_SCALE)
DISPLAY_H = int(A4_HEIGHT_PX * DISPLAY_SCALE)

HANDLE_SIZE = 8

# --- Data storage ---
template_images = []
template_texts = []
updated_images = []
current_index = 0
canvas = None
root = None
info_label = None

selected = {
    "id": None,
    "handle_id": None,
    "start_x": 0,
    "start_y": 0,
    "resize": False
}

bg_tk_img = None


# ---------------- CONVERSION -----------------
def to_display(x):
    return x * DISPLAY_SCALE

def to_real(x_disp):
    return x_disp / DISPLAY_SCALE


# ---------------- BOX MANAGEMENT -----------------
def draw_current_box():
    """Draws the current bounding box and handle."""
    global selected

    if current_index >= len(template_images):
        return

    entry = template_images[current_index]

    # remove previous
    if selected["id"]: canvas.delete(selected["id"])
    if selected["handle_id"]: canvas.delete(selected["handle_id"])

    x_tl = entry['x_disp']
    y_tl = entry['y_disp']
    w = entry['w_disp']
    h = entry['h_disp']

    # Draw box (integer canvas coords only)
    box_id = canvas.create_rectangle(
        int(x_tl), int(y_tl),
        int(x_tl + w), int(y_tl + h),
        outline="blue", width=2
    )
    selected["id"] = box_id

    # Draw handle
    half = HANDLE_SIZE // 2
    hx = int(x_tl + w)
    hy = int(y_tl + h)
    handle_id = canvas.create_rectangle(
        hx - half, hy - half, hx + half, hy + half,
        fill="black", outline="black"
    )
    selected["handle_id"] = handle_id
    canvas.tag_raise(handle_id)

    info_label.config(text=f"{current_index+1}/{len(template_images)} : {entry['file']}")


def update_box(x_tl, y_tl, w, h):
    entry = template_images[current_index]
    entry.update({"x_disp": x_tl, "y_disp": y_tl, "w_disp": w, "h_disp": h})

    # Update visuals
    canvas.coords(selected["id"],
                  int(x_tl), int(y_tl),
                  int(x_tl + w), int(y_tl + h))

    half = HANDLE_SIZE // 2
    canvas.coords(selected["handle_id"],
                  int(x_tl + w - half), int(y_tl + h - half),
                  int(x_tl + w + half), int(y_tl + h + half))


# ---------------- MOUSE EVENTS -----------------
# def on_mouse_down(event):
#     clicked = canvas.find_closest(event.x, event.y)[0]
#     if clicked == selected["handle_id"]:
#         selected.update({"resize": True})
#     elif clicked == selected["id"]:
#         selected.update({"resize": False})
#     else:
#         selected.update({"id": None})
#         return

#     selected["start_x"], selected["start_y"] = event.x, event.y

def on_mouse_down(event):
    clicked = canvas.find_closest(event.x, event.y)[0]

    # If clicked handle → resize mode
    if clicked == selected["handle_id"]:
        selected.update({
            "resize": True,
            "start_x": event.x,
            "start_y": event.y
        })
        return

    # If clicked inside active box → move mode
    if clicked == selected["id"]:
        selected.update({
            "resize": False,
            "start_x": event.x,
            "start_y": event.y
        })
        return

    # Otherwise → ignore click, keep selection
    # DO NOT remove selected["id"] anymore
    return



def on_mouse_move(event):
    if not selected["id"]:
        return

    dx = event.x - selected["start_x"]
    dy = event.y - selected["start_y"]
    entry = template_images[current_index]

    if selected["resize"]:
        # width/height change, top-left fixed
        new_w = max(15, entry['w_disp'] + dx)
        new_h = max(15, entry['h_disp'] + dy)
        update_box(entry['x_disp'], entry['y_disp'], new_w, new_h)
    else:
        # move entire box
        update_box(entry['x_disp'] + dx, entry['y_disp'] + dy,
                   entry['w_disp'], entry['h_disp'])

    selected["start_x"], selected["start_y"] = event.x, event.y


def on_mouse_up(event):
    selected.update({"resize": False})


# ---------------- FILE LOGIC -----------------
def load_template_and_image():
    global template_images, template_texts, current_index, bg_tk_img

    # Load JSON
    json_path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
    if not json_path: return
    with open(json_path, 'r') as f:
        data = json.load(f)

    template_images = data.get("images", [])
    template_texts = data.get("texts", [])

    # Convert all real coordinates to display float once
    for entry in template_images:
        entry["x_disp"] = to_display(entry["x"])
        entry["y_disp"] = to_display(entry["y"])
        entry["w_disp"] = to_display(entry["width"])
        entry["h_disp"] = to_display(entry["height"])

    # Load background
    img_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if not img_path: return

    img = Image.open(img_path).convert('RGB')
    img = img.resize((DISPLAY_W, DISPLAY_H))
    bg_tk_img = ImageTk.PhotoImage(img)

    canvas.delete("all")
    canvas.create_image(DISPLAY_W // 2, DISPLAY_H // 2, image=bg_tk_img)

    current_index = 0
    draw_current_box()


def next_item():
    global current_index

    if current_index >= len(template_images):
        messagebox.showinfo("Done", "All items fixed.")
        return

    entry = template_images[current_index]

    # Convert back preserving precision but rounding
    updated_images.append({
        "index": entry["index"],
        "file": entry["file"],
        "x": round(to_real(entry["x_disp"])),
        "y": round(to_real(entry["y_disp"])),
        "width": round(to_real(entry["w_disp"])),
        "height": round(to_real(entry["h_disp"]))
    })

    print("Inside next item", entry["index"], entry["file"])
    print(updated_images)

    current_index += 1
    if current_index < len(template_images):
        draw_current_box()
    else:
        canvas.delete(selected["id"])
        canvas.delete(selected["handle_id"])
        messagebox.showinfo("Complete", "Click Save New Template")

def save_course(data, s):
    base_name = "Course"
    w = data['width']
    h = data['height']
    x = data['x']
    y = data['y']
    factors = sf.COURSE_FACTORS
    if base_name.lower() in s:
        base_name += str(data["index"])

    # Course has 2 parts: bubbles and text which will be saved separately
    updated_data = []
    updated_bubbles = {"fieldType" : "QTYPE_INT",
                        "fieldLabels": [
                            "c1..4"
                        ],
                        "bubblesGap": int(factors["bubble_gap"]*w + factors["bubble_h"]*h),
                        "labelsGap": int(factors["label_gap"]*w),
                        "origin": [
                            int(x + factors["omr_x"]*w),
                            int(y + factors["omr_y"]*h)
                        ],
                        "bubbleDimensions": [
                            int(factors["bubble_w"]*w),
                            int(factors["bubble_h"]*h)
                        ]
                    }
    updated_text1 = {"fieldType": "TEXT_BOX",
                        "fieldLabels": [
                            "course_code1..2"
                        ],
                        "origin" : [
                            int(x + factors["box_x"]*w),
                            int(y + factors["box_y"]*h)
                        ],
                        "height" : int(factors["box_h"]*h),
                        "length" : int(factors["box_w"]*w)
                    }
    
    updated_text2 = {"fieldType": "NUM_BOX",
                        "fieldLabels": [
                            "course_num1..4"
                        ],
                        "origin" : [
                            int(x + factors["box_x"]*w + 2*factors["box_w"]*w),
                            int(y + factors["box_y"]*h)
                        ],
                        "height" : int(factors["box_h"]*h),
                        "length" : int(factors["box_w"]*w)
                    }
    
    updated_data.append([base_name + "_text1", base_name + "_text2", base_name + "_bubbles"])
    updated_data.append([updated_text1, updated_text2, updated_bubbles])
    
    custom_labels = {"Course_text": [
                            "course_code1..2",
                            "course_num1..4"
                        ]}
    return updated_data, custom_labels

def save_roll_no(data, s):
    base_name = "Roll_number"
    w = data['width']
    h = data['height']
    x = data['x']
    y = data['y']
    factors = sf.ROLL_NO_FACTORS
    if base_name.lower() in s:
        base_name += str(data["index"])

    # Roll number has 6 parts: text and bubbles - (Year, Program, Number) which will be saved separately
    updated_data = []
    updated_bubbles_yr = {"fieldType" : "QTYPE_INT",
                            "fieldLabels": [
                                "yb1..2"
                            ],
                            "bubblesGap": int(factors["bubble_gap"]*w + factors["bubble_h"]*h),
                            "labelsGap": int(factors["label_gap"]*w),
                            "origin": [
                                int(x + factors["omr_x"]*w),
                                int(y + factors["omr_y"]*h)
                            ],
                            "bubbleDimensions": [
                                int(factors["bubble_w"]*w),
                                int(factors["bubble_h"]*h)
                            ]
                        }
    
    updated_bubbles_prg = {"bubblesGap": int(factors["bubble_gap"]*w + factors["bubble_h"]*h),
                            "bubbleValues": [
                                "B",
                                "M",
                                "X",
                                "D"
                            ],
                            "direction": "vertical",
                            "fieldLabels": [
                                "progb"
                            ],
                            "labelsGap": 0,
                            "origin": [
                                int(x + factors["omr_x"]*w + 2*(factors["box_w"]*w)),
                                int(y + factors["omr_y"]*h)
                            ],
                            "bubbleDimensions": [
                                int(factors["bubble_w"]*w),
                                int(factors["bubble_h"]*h)
                            ]
                        }
    
    updated_bubbles_num = {"fieldType" : "QTYPE_INT",
                            "fieldLabels": [
                                "nb1..4"
                            ],
                            "bubblesGap": int(factors["bubble_gap"]*w + factors["bubble_h"]*h),
                            "labelsGap": int(factors["label_gap"]*w),
                            "origin": [
                                int(x + factors["omr_x"]*w + 3*(factors["box_w"]*w)),
                                int(y + factors["omr_y"]*h)
                            ],
                            "bubbleDimensions": [
                                int(factors["bubble_w"]*w),
                                int(factors["bubble_h"]*h)
                            ]
                        }
    

    updated_text_yr = {"fieldType": "NUM_BOX",
                        "fieldLabels": [
                            "yt1..2"
                        ],
                        "origin" : [
                            int(x + factors["box_x"]*w),
                            int(y + factors["box_y"]*h)
                        ],
                        "height" : int(factors["box_h"]*h),
                        "length" : int(factors["box_w"]*w)
                    }
    
    updated_text_prg = {"fieldType": "TEXT_BOX",
                        "fieldLabels": [
                            "progt"
                        ],
                        "origin" : [
                            int(x + factors["box_x"]*w + 2*(factors["box_w"]*w)),
                            int(y + factors["box_y"]*h)
                        ],
                        "height" : int(factors["box_h"]*h),
                        "length" : int(factors["box_w"]*w)
                    }

    
    updated_text_num = {"fieldType": "NUM_BOX",
                        "fieldLabels": [
                            "nt1..4"
                        ],
                        "origin" : [
                            int(x + factors["box_x"]*w + 3*(factors["box_w"]*w)),
                            int(y + factors["box_y"]*h)
                        ],
                        "height" : int(factors["box_h"]*h),
                        "length" : int(factors["box_w"]*w)
                    }
    
    updated_data.append([base_name + "_omr_yr", base_name + "_omr_prg", base_name + "_omr_num", base_name + "_text_yr", base_name + "_text_prg", base_name + "_text_num"])
    l = [updated_bubbles_yr, updated_bubbles_prg, updated_bubbles_num, updated_text_yr, updated_text_prg, updated_text_num]
    updated_data.append(l)

    custom_labels = {"Roll_No_text": [
                            "yt1..2",
                            "progt",
                            "nt1..4"
                        ],
                    "Roll_No_omr": [
                            "yb1..2",
                            "progb",
                            "nb1..4"
                        ]
                    }
                        
    return updated_data, custom_labels

def save_name(data, s):
    base_name = "Name"
    w = data['width']
    h = data['height']
    x = data['x']
    y = data['y']
    factors = sf.NAME_FACTORS
    if base_name.lower() in s:
        base_name += str(data["index"])

    # Course has 2 parts: bubbles and text which will be saved separately
    updated_data = []
    updated_text = {"fieldType": "TEXT_BOX",
                        "fieldLabels": [
                            base_name + "1..25"
                        ],
                        "origin" : [
                            int(x + factors["box_x"]*w),
                            int(y + factors["box_y"]*h)
                        ],
                        "height" : int(factors["box_h"]*h),
                        "length" : int(factors["box_w"]*w)
                    }
    
    updated_data.append([base_name])
    updated_data.append([updated_text])
                        
    return updated_data, []

def save_qr(data, s):
    base_name = "QR"
    w = data['width']
    h = data['height']
    x = data['x']
    y = data['y']
    factors = sf.QR_FACTORS
    if base_name.lower() in s:
        base_name += str(data["index"])

    # Course has 2 parts: bubbles and text which will be saved separately
    updated_data = []
    updated_text = {"fieldType": "QR_CODE",
                        "fieldLabels": [
                            base_name + "_"
                        ],
                        "origin" : [
                            int(x + factors["box_x"]*w*0.9),
                            int(y + factors["box_y"]*h*0.9)
                        ],
                        "height" : int(factors["box_h"]*h*1.2),
                        "length" : int(factors["box_w"]*w*1.2)
                    }
    
    updated_data.append([base_name])
    updated_data.append([updated_text])
                        
    return updated_data, []

def save_marks(data, s):
    base_name = "Marks"
    w = data['width']
    h = data['height']
    x = data['x']
    y = data['y']
    factors = sf.ROLL_NO_FACTORS
    if base_name.lower() in s:
        base_name += str(data["index"])

    # Roll number has 6 parts: text and bubbles - (Year, Program, Number) which will be saved separately
    updated_data = []
    updated_bubbles_1 = {"fieldType" : "QTYPE_INT",
                            "fieldLabels": [
                                "mb1..3"
                            ],
                            "bubblesGap": int(factors["bubble_gap"]*w + factors["bubble_h"]*h),
                            "labelsGap": int(factors["label_gap"]*w),
                            "origin": [
                                int(x + factors["omr_x"]*w),
                                int(y + factors["omr_y"]*h)
                            ],
                            "bubbleDimensions": [
                                int(factors["bubble_w"]*w),
                                int(factors["bubble_h"]*h)
                            ]
                        }
    
    updated_bubbles_2 = {"fieldType" : "QTYPE_INT",
                            "fieldLabels": [
                                "mdb1..2"
                            ],
                            "bubblesGap": int(factors["bubble_gap"]*w + factors["bubble_h"]*h),
                            "labelsGap": int(factors["label_gap"]*w),
                            "origin": [
                                int(x + factors["omr_x"]*w + 4*(factors["box_w"]*w)),
                                int(y + factors["omr_y"]*h)
                            ],
                            "bubbleDimensions": [
                                int(factors["bubble_w"]*w),
                                int(factors["bubble_h"]*h)
                            ]
                        }
    

    updated_text_1 = {"fieldType": "NUM_BOX",
                        "fieldLabels": [
                            "mt1..3"
                        ],
                        "origin" : [
                            int(x + factors["box_x"]*w),
                            int(y + factors["box_y"]*h)
                        ],
                        "height" : int(factors["box_h"]*h),
                        "length" : int(factors["box_w"]*w)
                    }
    
    updated_text_2 = {"fieldType": "NUM_BOX",
                        "fieldLabels": [
                            "mdt1..2"
                        ],
                        "origin" : [
                            int(x + factors["box_x"]*w + 4*(factors["box_w"]*w)),
                            int(y + factors["box_y"]*h)
                        ],
                        "height" : int(factors["box_h"]*h),
                        "length" : int(factors["box_w"]*w)
                    }
    
    
    updated_data.append([base_name + "_omr_1", base_name + "_omr_2", base_name + "_text_1", base_name + "_text_2"])
    l = [updated_bubbles_1, updated_bubbles_2, updated_text_1, updated_text_2]
    updated_data.append(l)

    custom_labels = {"Marks_text": [
                            "mt1..3",
                            "mdt1..2"
                        ],
                    "Marks_omr": [
                            "mb1..3",
                            "mdb1..2"
                        ]
                    }
                        
    return updated_data, custom_labels


def save_omr_template(updated_images):
    print("Entered save omr template")
    print(updated_images[3])
    existing_fields = set()
    field_blocks = []
    custom_labels_list = []
    for i in updated_images:

        # All the save_<> functions return 2 entries
        # The first entry is a list of field blocks and the second is there if a custom label is needed

        field_name = i["file"][:-4].lower()
        if field_name == "course":
            print("1 course")
            updated_data, custom_labels = save_course(i, existing_fields)
        elif field_name == "roll_no":
            print("2 roll")
            updated_data, custom_labels = save_roll_no(i, existing_fields)
        elif field_name == "marks":
            print("3 marks")
            updated_data, custom_labels = save_marks(i, existing_fields)
        elif field_name == "name":
            print("4 name")
            updated_data, custom_labels = save_name(i, existing_fields)
        elif field_name == "qr":
            print("5 qr")
            updated_data, custom_labels = save_qr(i, existing_fields)

        existing_fields.add(field_name)

        field_blocks.append(updated_data)
        if len(custom_labels) > 0:
            custom_labels_list.append(custom_labels)
    
    final_template = {"pageDimensions": [
                        2481,
                        3507
                    ],
                    "bubbleDimensions": [
                        40,
                        35
                    ],
                    "customLabels": {},
                    "fieldBlocks": {},
                    "preProcessors": [
                        {
                        "name": "CropPage",
                        "options": {
                            "morphKernel": [
                            5,
                            5
                            ]
                        }
                        }
                    ]}
    print("Field Blocks : ", field_blocks)
    for names, blocks in field_blocks:
        for i in range(len(names)):
            print(names)
            name = names[i]
            block = blocks[i]
            final_template["fieldBlocks"][name] = block

    for d in custom_labels_list:
        for k in d:
            final_template["customLabels"][k] = d[k]

    out_path = filedialog.asksaveasfilename(
        defaultextension="_omr.json",
        filetypes=[("JSON", "*.json")]
    )
    if not out_path: return

    with open(out_path, "w") as f:
        json.dump(final_template, f, indent=2)

    messagebox.showinfo("Saved OMR template ", os.path.basename(out_path))
    
    return -1


def save_new_template():
    if len(updated_images) != len(template_images):
        messagebox.showwarning("Warning", "Finish updating all items first")
        return
    
    # Saving the template in the way OMRChecker expects it
    save_omr_template(updated_images)

    # out_json = {
    #     "images": updated_images,
    #     "texts": template_texts  # untouched
    # }

    # out_path = filedialog.asksaveasfilename(
    #     defaultextension=".json",
    #     filetypes=[("JSON", "*.json")]
    # )
    # if not out_path: return

    # with open(out_path, "w") as f:
    #     json.dump(out_json, f, indent=2)

    # messagebox.showinfo("Saved", os.path.basename(out_path))


# ---------------- GUI SETUP -----------------
def init_gui():
    global root, canvas, info_label

    root = tk.Tk()
    root.title("Template Realignment Tool (High-Precision)")

    toolbar = tk.Frame(root, bg="#eee")
    toolbar.pack(side="top", fill="x")

    tk.Button(toolbar, text="Load Template/Image", command=load_template_and_image).pack(side="left")
    info_label = tk.Label(toolbar, text="Load Template JSON", width=50)
    info_label.pack(side="left", padx=10)
    tk.Button(toolbar, text="Next Item >>", command=next_item).pack(side="right")
    tk.Button(toolbar, text="Save New Template", command=save_new_template).pack(side="right")

    canvas = tk.Canvas(root, width=DISPLAY_W, height=DISPLAY_H, bg="white")
    canvas.pack(expand=True)

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    root.mainloop()


if __name__ == "__main__":
    init_gui()
