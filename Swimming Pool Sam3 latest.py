# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:44:06 2026
@author: arind
"""
import torch
import os
import glob
#################################### For Image ####################################
from PIL import ImageDraw
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import warnings

MAX_IMAGES    = 10

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Config ───────────────────────────────────────────────────────────────────
INPUT_FOLDER  = r"C:\Users\arind\Downloads\Pooldata\train\datasets\Galicia"
OUTPUT_SUFFIX = "_processedv2"          # processed images saved as <name>_processed.jpg
#PROMPT        = "swimming pool, backyard pool, outdoor pool"
PROMPT        = "Pool"
#PROMPT        = "House"

# ──────────────────────────────────────────────────────────────────────────────

# Load the model ONCE (outside the loop for efficiency)
model     = build_sam3_image_model()
processor = Sam3Processor(model)

# Gather all JPG files (case-insensitive)
jpg_files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) + \
            glob.glob(os.path.join(INPUT_FOLDER, "*.JPG")) + \
            glob.glob(os.path.join(INPUT_FOLDER, "*.jpeg")) + \
            glob.glob(os.path.join(INPUT_FOLDER, "*.JPEG"))

# De-duplicate in case the OS is case-insensitive
jpg_files = list(dict.fromkeys(jpg_files))

jpg_files = jpg_files[:MAX_IMAGES]

if not jpg_files:
    print(f"No JPG files found in: {INPUT_FOLDER}")
    exit()

print(f"Found {len(jpg_files)} image(s). Processing…\n")

results = []   # list of (original_filename, processed_filename) for HTML

for img_path in jpg_files:
    print(f"  Processing: {os.path.basename(img_path)}")

    # ── Image processing code (UNCHANGED) ────────────────────────────────────
    image = Image.open(img_path)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=PROMPT)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    draw = ImageDraw.Draw(image)
    for box in boxes:
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    # ─────────────────────────────────────────────────────────────────────────

    # Save processed image alongside the original
    base, ext = os.path.splitext(os.path.basename(img_path))
    out_filename = base + OUTPUT_SUFFIX + ext
    out_path     = os.path.join(INPUT_FOLDER, out_filename)
    image.save(out_path)
    print(f"    → Saved: {out_filename}")

    results.append((os.path.basename(img_path), out_filename))

print(f"\nDone. {len(results)} image(s) processed.")

# ─── Generate HTML comparison viewer ──────────────────────────────────────────
html_rows = ""
for orig, proc in results:
    html_rows += f"""
        <tr>
            <td class="label">{orig}</td>
            <td><img src="{orig}"  alt="Original: {orig}"></td>
            <td><img src="{proc}" alt="Processed: {proc}"></td>
        </tr>"""

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Swimming Pool Detection – Results</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0f1117;
            color: #e0e0e0;
            padding: 30px 20px;
        }}

        h1 {{
            text-align: center;
            font-size: 1.8rem;
            margin-bottom: 6px;
            color: #ffffff;
            letter-spacing: 1px;
        }}

        .subtitle {{
            text-align: center;
            color: #888;
            font-size: 0.95rem;
            margin-bottom: 30px;
        }}

        table {{
            width: 100%;
            max-width: 1400px;
            margin: 0 auto;
            border-collapse: collapse;
        }}

        thead th {{
            background: #1e2130;
            padding: 14px 16px;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #aaa;
            border-bottom: 2px solid #2a2d3e;
        }}

        thead th:first-child {{ text-align: left; width: 22%; }}
        thead th:not(:first-child) {{ text-align: center; width: 39%; }}

        tbody tr {{
            border-bottom: 1px solid #1e2130;
            transition: background 0.2s;
        }}
        tbody tr:hover {{ background: #161923; }}

        td {{
            padding: 14px 16px;
            vertical-align: middle;
        }}

        td.label {{
            font-size: 0.82rem;
            color: #9ab;
            word-break: break-all;
        }}

        td img {{
            display: block;
            width: 100%;
            max-width: 560px;
            height: auto;
            border-radius: 6px;
            border: 1px solid #2a2d3e;
            margin: 0 auto;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s;
        }}

        td img:hover {{
            transform: scale(1.02);
            box-shadow: 0 6px 24px rgba(0,0,0,0.6);
        }}

        /* Light-box */
        #lightbox {{
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.88);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            cursor: zoom-out;
        }}
        #lightbox.active {{ display: flex; }}
        #lightbox img {{
            max-width: 92vw;
            max-height: 92vh;
            border-radius: 8px;
            box-shadow: 0 8px 40px rgba(0,0,0,0.9);
        }}

        .badge {{
            display: inline-block;
            background: #e53935;
            color: #fff;
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 12px;
            margin-left: 8px;
            vertical-align: middle;
            font-weight: 600;
            letter-spacing: 0.05em;
        }}
    </style>
</head>
<body>

    <h1>🏊 Swimming Pool Detection <span class="badge">SAM3</span></h1>
    <p class="subtitle">Prompt: "{PROMPT}" &nbsp;·&nbsp; {len(results)} image(s) processed &nbsp;·&nbsp; Folder: {INPUT_FOLDER}</p>

    <table>
        <thead>
            <tr>
                <th>Filename</th>
                <th>Original</th>
                <th>Processed (Detections)</th>
            </tr>
        </thead>
        <tbody>{html_rows}
        </tbody>
    </table>

    <!-- Light-box overlay -->
    <div id="lightbox">
        <img id="lightbox-img" src="" alt="">
    </div>

    <script>
        const lb    = document.getElementById('lightbox');
        const lbImg = document.getElementById('lightbox-img');

        document.querySelectorAll('td img').forEach(img => {{
            img.addEventListener('click', () => {{
                lbImg.src = img.src;
                lb.classList.add('active');
            }});
        }});

        lb.addEventListener('click', () => lb.classList.remove('active'));
    </script>
</body>
</html>
"""

html_path = os.path.join(INPUT_FOLDER, "resultsv2.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\nHTML viewer saved → {html_path}")
print("Open results.html in your browser to compare original vs processed images.")
