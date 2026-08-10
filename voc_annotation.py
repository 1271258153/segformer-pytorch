import os

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#
#   当前目录结构：
#   VOCdevkit/VOC2007/
#   ├── JPEGImages/{train,val,test}/*.jpg
#   └── SegmentationClass/{train,val,test}/*.png
#
#   生成的 train.txt / val.txt / test.txt 内容形如：
#   train/003
#   val/001
#-------------------------------------------------------#
VOCdevkit_path  = 'VOCdevkit'
splits          = ['train', 'val', 'test']

def list_split_names(seg_root, split):
    split_dir = os.path.join(seg_root, split)
    if not os.path.isdir(split_dir):
        print("警告：未找到目录 %s，跳过。" % split_dir)
        return []

    names = []
    for seg in sorted(os.listdir(split_dir)):
        if seg.lower().endswith('.png'):
            stem = os.path.splitext(seg)[0]
            names.append('%s/%s' % (split, stem))
    return names

def write_txt(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

if __name__ == "__main__":
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    jpgfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/JPEGImages')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    os.makedirs(saveBasePath, exist_ok=True)

    print("Generate txt in ImageSets.")
    split_names = {}
    for split in splits:
        split_names[split] = list_split_names(segfilepath, split)
        print("%s size: %d" % (split, len(split_names[split])))

    # 检查标签与原图是否一一对应
    missing = []
    for split in splits:
        for name in split_names[split]:
            jpg_path = os.path.join(jpgfilepath, name + '.jpg')
            png_path = os.path.join(segfilepath, name + '.png')
            if not os.path.exists(png_path):
                missing.append(png_path)
            if not os.path.exists(jpg_path):
                missing.append(jpg_path)
    if missing:
        print("以下文件缺失（仅展示前20个）：")
        for p in missing[:20]:
            print("  " + p)
        raise FileNotFoundError("共有 %d 个文件缺失，请检查目录。" % len(missing))

    write_txt(os.path.join(saveBasePath, 'train.txt'), split_names['train'])
    write_txt(os.path.join(saveBasePath, 'val.txt'), split_names['val'])
    write_txt(os.path.join(saveBasePath, 'test.txt'), split_names['test'])
    write_txt(
        os.path.join(saveBasePath, 'trainval.txt'),
        split_names['train'] + split_names['val']
    )
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    all_names = split_names['train'] + split_names['val'] + split_names['test']
    classes_nums = np.zeros([256], np.int64)
    for name in tqdm(all_names):
        png_file_name = os.path.join(segfilepath, name + '.png')
        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。" % (name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。")

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")
