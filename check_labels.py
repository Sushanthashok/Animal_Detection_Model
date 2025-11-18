# check_labels.py
import os
from PIL import Image

root = "final/data"   # <- adjust if needed
splits = ["train", "valid", "test"]

errors = []
for s in splits:
    img_dir = os.path.join(root, s, "images")
    lbl_dir = os.path.join(root, s, "labels")
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        errors.append(f"Missing {s}/images or {s}/labels")
        continue
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    for img in imgs:
        name = os.path.splitext(img)[0]
        lbl_file = os.path.join(lbl_dir, name + ".txt")
        if not os.path.exists(lbl_file):
            errors.append(f"Missing label for {img} in {s}")
            continue
        w,h = Image.open(os.path.join(img_dir,img)).size
        with open(lbl_file,'r') as fh:
            for i,line in enumerate(fh,1):
                parts=line.strip().split()
                if len(parts)!=5:
                    errors.append(f"{lbl_file}:{i} malformed")
                    continue
                _,xc,yc,w_n,h_n = parts
                try:
                    vals = list(map(float, (xc,yc,w_n,h_n)))
                except:
                    errors.append(f"{lbl_file}:{i} non-float")
                    continue
                if not all(0.0 <= v <= 1.0 for v in vals):
                    errors.append(f"{lbl_file}:{i} normalized out of range")
if errors:
    print("Found label issues:")
    for e in errors[:50]:
        print("-",e)
else:
    print("Labels look OK.")
