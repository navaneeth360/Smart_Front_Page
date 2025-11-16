import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import json

# --- DEFAULT PATH FOR LOADING IMAGES ---
# !!! IMPORTANT: CHANGE THIS TO A VALID PATH ON YOUR SYSTEM FOR DEFAULT OPERATION !!!
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOAD_IMAGE_DIR   = os.path.join(SCRIPT_DIR, "Field_images")
DEFAULT_SAVE_LAYOUT_DIR  = os.path.join(SCRIPT_DIR, "Templates\\json")
DEFAULT_EXPORT_LAYOUT_DIR = os.path.join(SCRIPT_DIR, "Templates\\pdf")

for folder in [DEFAULT_SAVE_LAYOUT_DIR, DEFAULT_EXPORT_LAYOUT_DIR]:
    os.makedirs(folder, exist_ok=True)
# ---------------------------------------

# --- A4 constants ---
DPI = 300
SCREEN_DPI = 66
A4_WIDTH_INCH = 8.27
A4_HEIGHT_INCH = 11.69
A4_WIDTH_PX = int(A4_WIDTH_INCH * DPI)
A4_HEIGHT_PX = int(A4_HEIGHT_INCH * DPI)

# --- Display scaling ---
DISPLAY_SCALE = 0.3
DISPLAY_W = int(A4_WIDTH_PX * DISPLAY_SCALE)
DISPLAY_H = int(A4_HEIGHT_PX * DISPLAY_SCALE)

# --- Font for Text Export (NEW) ---
try:
    DEFAULT_EXPORT_FONT = ImageFont.truetype("arial.ttf", 30)
except IOError:
    DEFAULT_EXPORT_FONT = ImageFont.load_default()
# -----------------------------------

root = tk.Tk()
root.title("A4 Layout Tool")

toolbar = tk.Frame(root, bg="#f0f0f0", padx=5, pady=5)
toolbar.pack(side='top', fill='x')

canvas_frame = tk.Frame(root)
canvas_frame.pack(side='bottom', fill='both', expand=True)

canvas = tk.Canvas(canvas_frame, width=DISPLAY_W, height=DISPLAY_H, bg='white')
canvas.pack(expand=True)

hint_label = tk.Label(toolbar, text="🖼️ Add images, drag to move, resize using the black square handle. ✍️ Add text from toolbar.")
hint_label.pack(side='left', padx=10)

# Data structure:
# [pil_image, image_canvas_id, border_rect_id, handle_id, x_disp_center, y_disp_center, w_disp, h_disp, path]
placed_images = []
placed_texts = []  # List of canvas IDs for text items

# selection state
selected = {
    "id": None,        # canvas id of clicked item (image/rect/handle)
    "start_x": 0,
    "start_y": 0,
    "resize": False,   # True when resizing (via handle)
    "dragging_handle": False
}
outline_rect = None
mode = "select"

# Snap & grid
snap_on = False
grid_size = 10  # default 10 px

# handle appearance
HANDLE_SIZE = 8  # display pixels (square side)


def snap_value(v):
    """Snap a single display coordinate/size to grid if snap_on."""
    if not snap_on:
        return int(round(v))
    return int(round(v / grid_size) * grid_size)


def add_image(path=None, real_x=None, real_y=None, real_w=None, real_h=None):
    if not path or not os.path.exists(path):
        path = filedialog.askopenfilename(
                initialdir=DEFAULT_LOAD_IMAGE_DIR,
                filetypes=[("Image files", "*.jpg *.png *.jpeg")]
            )
        if not path:
            return

    img = Image.open(path)

    if real_w and real_h:
        display_w = int(real_w * DISPLAY_SCALE)
        display_h = int(real_h * DISPLAY_SCALE)
        # Resize real image to requested real dims for better quality when scaling down to display
        resized = img.copy().resize((real_w, real_h))
        display_img = resized.copy().resize((display_w, display_h))
    else:
        display_img = img.copy()
        display_img.thumbnail((400, 400))

    tk_img = ImageTk.PhotoImage(display_img)

    # Convert real top-left corner coordinates to display center coordinates (existing behavior)
    if real_x is not None and real_y is not None:
        x_disp_center = real_x * DISPLAY_SCALE + display_img.width / 2
        y_disp_center = real_y * DISPLAY_SCALE + display_img.height / 2
    else:
        x_disp_center = DISPLAY_W // 2
        y_disp_center = DISPLAY_H // 2

    # Apply snapping for placement if snap_on
    x_disp_center = snap_value(x_disp_center)
    y_disp_center = snap_value(y_disp_center)

    img_id = canvas.create_image(x_disp_center, y_disp_center, image=tk_img, anchor='center')
    # Keep a reference so Tk doesn't garbage-collect
    canvas.image = getattr(canvas, "image", []) + [tk_img]

    # Create thin black border rectangle (display coords)
    rect_id = canvas.create_rectangle(
        x_disp_center - display_img.width / 2, y_disp_center - display_img.height / 2,
        x_disp_center + display_img.width / 2, y_disp_center + display_img.height / 2,
        outline="black", width=1
    )

    handle_id = None

    placed_images.append([img, img_id, rect_id, handle_id, x_disp_center, y_disp_center, display_img.width, display_img.height, path])


def find_image_by_id(cid):
    for item in placed_images:
        if item[1] == cid or item[2] == cid or (item[3] == cid if item[3] is not None else False):
            return item
    return None


def create_handle_for(item):
    """Create (or recreate) the small square handle at bottom-right. Returns handle id."""
    if item[3]:
        try:
            canvas.delete(item[3])
        except Exception:
            pass
        item[3] = None

    x_center, y_center, w, h = item[4], item[5], item[6], item[7]
    half = HANDLE_SIZE // 2
    # bottom-right coordinates in display (since x_center,y_center are centers)
    br_x = x_center + w / 2
    br_y = y_center + h / 2
    hx1 = br_x - half
    hy1 = br_y - half
    hx2 = br_x + half
    hy2 = br_y + half
    hid = canvas.create_rectangle(hx1, hy1, hx2, hy2, fill="black", outline="black")
    # Make sure handle is above border & image
    canvas.tag_raise(hid, item[1])
    canvas.tag_raise(hid, item[2])
    item[3] = hid
    return hid


def remove_handle_for(item):
    if item and item[3]:
        try:
            canvas.delete(item[3])
        except Exception:
            pass
        item[3] = None


def draw_outline(item):
    """Draw temporary blue dashed outline for selected image (visual cue)."""
    global outline_rect
    if outline_rect:
        canvas.delete(outline_rect)
        outline_rect = None
    if item:
        x, y, w, h = item[4], item[5], item[6], item[7]
        outline_rect = canvas.create_rectangle(
            x - w / 2, y - h / 2, x + w / 2, y + h / 2,
            outline="blue", dash=(4, 2), width=2
        )
        # keep outline above image rect
        canvas.tag_raise(outline_rect, item[3] if item[3] else item[2])


def on_mouse_down(event):
    global mode, selected
    if mode == "add_text":
        text_content = simpledialog.askstring("Add Text", "Enter text:")
        if text_content:
            # font_size = simpledialog.askinteger("Font Size", "Enter font size (display px):", initialvalue=12, minvalue=8, maxvalue=100)
            # tid = canvas.create_text(event.x, event.y, text=text_content, anchor='nw', font=("Arial", font_size))
            font_size = simpledialog.askinteger("Font Size", "Enter font size (display px):",
                                    initialvalue=12, minvalue=8, maxvalue=100)

            # If user cancels → font_size = None
            if font_size is None:
                mode = "select"
                return

            tid = canvas.create_text(event.x, event.y, text=text_content,
                                    anchor='nw', font=("Arial", font_size))
            placed_texts.append(tid)
        mode = "select"
        return

    clicked = canvas.find_closest(event.x, event.y)
    cid = clicked[0] if clicked else None

    if cid:
        item = find_image_by_id(cid)
        if item:
            # If the click was on the handle, start a resize
            if item[3] == cid:
                # Keep selected id as the image id so find_image_by_id works later
                selected.update({"id": item[1], "start_x": event.x, "start_y": event.y, "resize": True, "dragging_handle": True})
                draw_outline(item)
                return

            # Otherwise, we clicked the image or its rectangle - start move or potentially resize if click near bottom-right
            ix, iy, iw, ih = item[4], item[5], item[6], item[7]
            br_x = ix + iw / 2
            br_y = iy + ih / 2
            if abs(event.x - br_x) <= HANDLE_SIZE * 1.5 and abs(event.y - br_y) <= HANDLE_SIZE * 1.5:
                if not item[3]:
                    create_handle_for(item)
                selected.update({"id": item[1], "start_x": event.x, "start_y": event.y, "resize": True, "dragging_handle": True})
                draw_outline(item)
                return
            else:
                # clicked on the image - begin dragging (move)
                selected.update({"id": item[1], "start_x": event.x, "start_y": event.y, "resize": False, "dragging_handle": False})
                draw_outline(item)
                if not item[3]:
                    create_handle_for(item)
                if item[3]:
                    canvas.tag_raise(item[3])
                return
        else:
            # clicked something else (maybe text). If text, allow dragging
            if cid in placed_texts:
                selected.update({"id": cid, "start_x": event.x, "start_y": event.y, "resize": False, "dragging_handle": False})
                draw_outline(None)  # no outline for text
                return
            else:
                # clicked non-image canvas object: DO NOT clear selection (fixes bug)
                # simply ignore click so selection stays active for resizing later
                return
    else:
        # clicked empty area: DO NOT deselect everything (fixes bug where resize becomes disabled)
        # If you prefer explicit deselect, add a toolbar button to deselect.
        return


def on_mouse_move(event):
    if selected["id"] is None:
        return

    # selected["id"] is the image canvas id (we stored item[1])
    item = find_image_by_id(selected["id"])
    if not item:
        # maybe it's text
        if selected["id"] in placed_texts:
            dx = event.x - selected["start_x"]
            dy = event.y - selected["start_y"]
            canvas.move(selected["id"], dx, dy)
            selected["start_x"], selected["start_y"] = event.x, event.y
        return

    img_id, rect_id, handle_id = item[1], item[2], item[3]

    if selected["resize"]:
        # Resizing via handle or bottom-right area
        dx = event.x - selected["start_x"]
        dy = event.y - selected["start_y"]

        # Compute new width/height in display coords (relative to previous stored display w,h)
        old_w = item[6]
        old_h = item[7]
        new_w = max(20, old_w + dx)
        new_h = max(20, old_h + dy)

        # Snap size to grid if snap_on
        if snap_on:
            new_w = max(20, int(round(new_w / grid_size) * grid_size))
            new_h = max(20, int(round(new_h / grid_size) * grid_size))

        # Compute top-left (in display coords) from old center & old size
        tl_x = item[4] - old_w / 2
        tl_y = item[5] - old_h / 2

        # New center so that top-left stays fixed
        new_center_x = tl_x + new_w / 2
        new_center_y = tl_y + new_h / 2

        # Resize the real PIL image to match new display size (exact)
        target_real_w = int(new_w / DISPLAY_SCALE)
        target_real_h = int(new_h / DISPLAY_SCALE)
        # guard
        target_real_w = max(1, target_real_w)
        target_real_h = max(1, target_real_h)
        resized = item[0].copy().resize((target_real_w, target_real_h))
        display_img = resized.copy().resize((int(new_w), int(new_h)))

        tk_img = ImageTk.PhotoImage(display_img)
        # Update canvas image (keep anchor=center as we store center coords)
        canvas.itemconfig(img_id, image=tk_img)
        canvas.image.append(tk_img)

        # Update stored display size and center so top-left remains same
        item[6], item[7] = float(new_w), float(new_h)
        item[4], item[5] = float(new_center_x), float(new_center_y)

        # Update rectangle coords (rectangle uses center coords)
        canvas.coords(rect_id,
                      item[4] - item[6] / 2, item[5] - item[7] / 2,
                      item[4] + item[6] / 2, item[5] + item[7] / 2)

        # Update handle position
        if handle_id:
            half = HANDLE_SIZE // 2
            hx1 = item[4] + item[6] / 2 - half
            hy1 = item[5] + item[7] / 2 - half
            hx2 = item[4] + item[6] / 2 + half
            hy2 = item[5] + item[7] / 2 + half
            canvas.coords(handle_id, hx1, hy1, hx2, hy2)
            canvas.tag_raise(handle_id)

        # Update the image canvas coords (center anchor)
        canvas.coords(img_id, item[4], item[5])

        # Update selected start so further motion continues resizing relative to last point
        selected["start_x"], selected["start_y"] = event.x, event.y

        draw_outline(item)

    else:
        # Moving the image (drag)
        dx = event.x - selected["start_x"]
        dy = event.y - selected["start_y"]

        # Move items visually
        canvas.move(img_id, dx, dy)
        canvas.move(rect_id, dx, dy)
        if handle_id:
            canvas.move(handle_id, dx, dy)

        # Update stored center
        item[4] += dx
        item[5] += dy

        # Snap center to grid if snap_on
        if snap_on:
            item[4] = snap_value(item[4])
            item[5] = snap_value(item[5])
            # After snapping center, move canvas items to align
            canvas.coords(img_id, item[4], item[5])
            canvas.coords(rect_id,
                          item[4] - item[6] / 2, item[5] - item[7] / 2,
                          item[4] + item[6] / 2, item[5] + item[7] / 2)
            if handle_id:
                half = HANDLE_SIZE // 2
                hx1 = item[4] + item[6] / 2 - half
                hy1 = item[5] + item[7] / 2 - half
                hx2 = item[4] + item[6] / 2 + half
                hy2 = item[5] + item[7] / 2 + half
                canvas.coords(handle_id, hx1, hy1, hx2, hy2)

        selected["start_x"], selected["start_y"] = event.x, event.y

        draw_outline(item)


def on_mouse_up(event):
    # Clear resizing state but keep selection and handle visible.
    selected.update({"resize": False, "dragging_handle": False})
    global outline_rect
    if outline_rect:
        canvas.delete(outline_rect)
        outline_rect = None


def export_layout():
    output = Image.new('RGB', (A4_WIDTH_PX, A4_HEIGHT_PX), 'white')
    draw = ImageDraw.Draw(output)

    # 1. Draw Images and Borders
    for img, img_id, rect_id, handle_id, x_disp, y_disp, w_disp, h_disp, path in placed_images:
        real_w = int(w_disp / DISPLAY_SCALE)
        real_h = int(h_disp / DISPLAY_SCALE)
        real_x = int(x_disp / DISPLAY_SCALE - real_w / 2)
        real_y = int(y_disp / DISPLAY_SCALE - real_h / 2)

        real_w = max(1, real_w)
        real_h = max(1, real_h)

        resized = img.copy().resize((real_w, real_h))
        output.paste(resized, (real_x, real_y))

        draw.rectangle([real_x, real_y, real_x + real_w, real_y + real_h], outline="black", width=3)

    # 2. Draw Text Elements
    for tid in placed_texts:
        text_content = canvas.itemcget(tid, 'text')
        font_string = canvas.itemcget(tid, 'font')

        try:
            font_parts = font_string.split()
            font_family = font_parts[0]
            font_size_disp = int(font_parts[1])
        except (IndexError, ValueError):
            font_family = "Arial"
            font_size_disp = 12

        x_disp, y_disp = canvas.coords(tid)

        real_x = int(x_disp / DISPLAY_SCALE)
        real_y = int(y_disp / DISPLAY_SCALE)

        real_font_size = max(8, int(font_size_disp * DPI / SCREEN_DPI))

        try:
            export_font = ImageFont.truetype(f"{font_family.lower()}.ttf", real_font_size)
        except IOError:
            export_font = DEFAULT_EXPORT_FONT.font_variant(size=real_font_size)
        except Exception:
            export_font = DEFAULT_EXPORT_FONT.font_variant(size=real_font_size)

        draw.text((real_x, real_y), text_content, font=export_font, fill=(0, 0, 0))

    save_path = filedialog.asksaveasfilename(
            initialdir=DEFAULT_EXPORT_LAYOUT_DIR,
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("JPEG", "*.jpg")]
        )
    
    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".pdf":
            output.save(save_path, "PDF", resolution=DPI)
        else:
            output.save(save_path, "JPEG", quality=95)


def save_layout_metadata():
    metadata = {}

    image_metadata = []
    for index, item in enumerate(placed_images):
        img, img_id, rect_id, handle_id, x_disp, y_disp, w_disp, h_disp, path = item

        real_w = int(w_disp / DISPLAY_SCALE)
        real_h = int(h_disp / DISPLAY_SCALE)
        real_x = int(x_disp / DISPLAY_SCALE - real_w / 2)
        real_y = int(y_disp / DISPLAY_SCALE - real_h / 2)

        filename = os.path.basename(path) if path else ""
        image_metadata.append({
            "index": index,
            "file": filename,
            "x": real_x,
            "y": real_y,
            "width": real_w,
            "height": real_h
        })
    metadata['images'] = image_metadata

    text_metadata = []
    for tid in placed_texts:
        text_content = canvas.itemcget(tid, 'text')
        font_string = canvas.itemcget(tid, 'font')
        x_disp, y_disp = canvas.coords(tid)

        real_x = int(x_disp / DISPLAY_SCALE)
        real_y = int(y_disp / DISPLAY_SCALE)

        text_metadata.append({
            "content": text_content,
            "x": real_x,
            "y": real_y,
            "font": font_string
        })
    metadata['texts'] = text_metadata

    file_path = filedialog.asksaveasfilename(
            initialdir=DEFAULT_SAVE_LAYOUT_DIR,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")]
        )
    if file_path:
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_layout_metadata():
    # 1. Prompt for JSON file
    # json_path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")], title="Select Layout JSON File")
    json_path = filedialog.askopenfilename(
            initialdir=DEFAULT_SAVE_LAYOUT_DIR,
            title="Select Layout JSON File",
            filetypes=[("JSON", "*.json")]
    )
    if not json_path:
        return

    # 2. Prompt for Image Folder Path
    image_folder = simpledialog.askstring(
        "Load Images",
        "Enter the folder path containing the template images:",
        initialvalue=DEFAULT_LOAD_IMAGE_DIR
    )
    if not image_folder:
        return

    if not os.path.isdir(image_folder):
        tk.messagebox.showerror("Error", f"Image folder not found: {image_folder}")
        return

    with open(json_path, 'r') as f:
        metadata = json.load(f)

    # Clear existing
    for _, img_id, rect_id, handle_id, *_ in placed_images:
        try:
            canvas.delete(img_id)
            canvas.delete(rect_id)
            if handle_id: canvas.delete(handle_id)
        except Exception:
            pass
    placed_images.clear()

    for tid in placed_texts:
        try:
            canvas.delete(tid)
        except Exception:
            pass
    placed_texts.clear()

    # Load Images
    image_data = metadata.get('images', [])
    for entry in image_data:
        full_image_path = os.path.join(image_folder, entry["file"])
        add_image(full_image_path, entry["x"], entry["y"], entry["width"], entry["height"])

    # Load Text
    text_data = metadata.get('texts', [])
    for entry in text_data:
        x_disp = int(entry['x'] * DISPLAY_SCALE)
        y_disp = int(entry['y'] * DISPLAY_SCALE)
        tid = canvas.create_text(x_disp, y_disp, text=entry["content"], anchor='nw', font=entry["font"])
        placed_texts.append(tid)


def enable_text_mode():
    global mode
    mode = "add_text"


def toggle_snap():
    global snap_on
    snap_on = not snap_on
    snap_btn.config(text=f"Snap: {'ON' if snap_on else 'OFF'}")


# Bindings
canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_move)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

# Buttons
tk.Button(toolbar, text="Add Image", command=add_image).pack(side='left', padx=5)
tk.Button(toolbar, text="Add Text", command=enable_text_mode).pack(side='left', padx=5)
tk.Button(toolbar, text="Export Layout", command=export_layout).pack(side='left', padx=5)
tk.Button(toolbar, text="Save Layout", command=save_layout_metadata).pack(side='left', padx=5)
tk.Button(toolbar, text="Load Layout", command=load_layout_metadata).pack(side='left', padx=5)

# Snap toggle button
snap_btn = tk.Button(toolbar, text=f"Snap: {'ON' if snap_on else 'OFF'}", command=toggle_snap)
snap_btn.pack(side='left', padx=5)

# Grid size label & quick change
def set_grid_size():
    global grid_size
    val = simpledialog.askinteger("Grid Size", "Enter grid size in pixels:", initialvalue=grid_size, minvalue=1, maxvalue=200)
    if val:
        grid_size = val
        grid_label.config(text=f"Grid: {grid_size}px")

grid_label = tk.Button(toolbar, text=f"Grid: {grid_size}px (change)", command=set_grid_size)
grid_label.pack(side='left', padx=5)

root.minsize(900, 700)
root.mainloop()
