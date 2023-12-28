在这个文件下放入自己数据
image.png
image_mask.png

所有的数据都是 原图.png
mask图像：只需要在 原图_mask.png

如果分类不一样，需要改一下dataset.py中__getitem__，将mask的label改一下，默认是个人项目需要
class ImageFold(data.Dataset):
    def __getitem__(self, idx):
