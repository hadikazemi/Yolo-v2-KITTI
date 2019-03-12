import glob
from skimage import io

addrs = glob.glob('/media/hadi/HHD 6TB/Datasets/Kitti/training/image_2/*')

for a in addrs:
    img = io.imread(a)
    print img.shape