import os
from PIL import Image

# JPEGImages 目录（支持 train/val/test 子目录）
img_dir = os.path.join("VOCdevkit", "VOC2007", "JPEGImages")

if not os.path.isdir(img_dir):
    raise FileNotFoundError("未找到目录: %s" % img_dir)

converted = 0
for root, _, files in os.walk(img_dir):
    for name in files:
        if not name.lower().endswith(".png"):
            continue
        png_path = os.path.join(root, name)
        jpg_name = os.path.splitext(name)[0] + ".jpg"
        jpg_path = os.path.join(root, jpg_name)

        img = Image.open(png_path).convert("RGB")
        img.save(jpg_path, quality=95)
        os.remove(png_path)
        converted += 1
        if converted % 500 == 0:
            print("已转换 %d 张..." % converted)

print("完成，共转换 %d 张 png -> jpg" % converted)
