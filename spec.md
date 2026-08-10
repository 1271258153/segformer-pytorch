## 训练自己的数据集
1. 将数据集拷贝到 `datasets/infrared_images/` 下
2. 将图片拷贝到 `VOCdevkit/VOC2007/JPEGImages/` 下，标签拷贝到 `VOCdevkit/VOC2007/SegmentationClass/` 下
3. 执行 `python png2jpg.py` ,将 JPEGImages 下的png图片转变为jpg格式
4. 执行 `python voc_annotation.py`, 生成txt标签文件
5. 将预训练权重放入 `model_data/` 下

### 训练
```bash
python train.py --config experiments/infrared_images/test.yaml && /usr/bin/shutdown
```

### 评估
TEST_SET 改为 `VOC2007/ImageSets/Segmentation/test.txt`
```bash
python get_miou.py --config experiments/infrared_images/test.yaml
```