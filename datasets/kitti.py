import torch
import torch.nn as nn
import functools
import torch.utils.data as data
import os
import numpy as np
import torchvision.transforms as transforms
import glob
from PIL import Image
from skimage import io
from random import shuffle
from utils.im_transform import imcv2_recolor
from skimage.transform import resize
import cv2
from .kitti_eval import kitti_eval
import pickle


class Kitti(data.Dataset):
    def __init__(self, root_path, stage='train', cfg=None):
        assert os.path.isdir(root_path), '%s is not a valid directory' % root_path

        # root_path = root_path.replace('/training', '/testing')
        self.root_path = root_path
        self.img_path = os.path.join(root_path, 'image_2/*.png')
        print 'data path: ', root_path

        self.img_path = glob.glob(self.img_path)

        # List all JPEG images
        if stage=='train':
            self.img_path = self.img_path[:6000]
        if stage=='val':
            self.img_path = self.img_path[6000:6481] #6481
        if stage=='test':
            self.img_path = self.img_path[6481:]

        self.image_indexes = [p.split('/')[-1].split('.')[0] for p in self.img_path]

        self.size = len(self.img_path)

        self.transform = transforms.Compose([transforms.Scale((366, 1230)), transforms.ToTensor()])
        self.cfg = cfg

        self.classes = ('Pedestrian', 'Truck', 'Car', 'Cyclist',
                        'Misc', 'Van', 'Sitting', 'Tram')

    def __getitem__(self, index):
        # img = io.imread(self.img_path[index % self.size])
        # img = resize(img, (366, 1230), preserve_range=True)
        img = cv2.imread(self.img_path[index % self.size])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1230, 366))
        ori_im = (img.copy()).astype(np.uint8)

        img = imcv2_recolor(img)
        img = np.moveaxis(img, 2, 0)
        img = torch.from_numpy(img).type(torch.FloatTensor)

        # img = Image.open(self.img_path[index % self.size]).convert('RGB')
        # img = self.transform(img)                                   # Apply the defined transform

        lbl_file = self.img_path[index % self.size].replace('.png', '.txt').replace('image_2', 'label_2')
        with open(lbl_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        gt_boxes = []
        gt_classes = []
        for c in content:
            bb = c.split(' ')
            if bb[0] in self.cfg.label_names:
                cc = self.cfg.label_names.index(bb[0])
            else:
                continue
            bb = map(int, map(float, bb[4:8]))
            gt_boxes.append(bb)
            gt_classes.append(cc)

        # img = img.type(torch.FloatTensor)

        gt_boxes = np.array(gt_boxes)
        gt_classes = np.array(gt_classes)

        return img, gt_boxes, gt_classes, ori_im

    def __len__(self):
        # Provides the size of the dataset
        return self.size

    def evaluate_detections(self, all_boxes, output_dir=None):
        self._write_kitti_results_file(all_boxes)
        self._do_python_eval(output_dir)
        for cls in self.classes:
            if cls == '__background__':
                continue
            filename = self._get_kitti_results_file_template().format(cls)
            os.remove(filename)

    def _write_kitti_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} KITTI results file'.format(cls))
            filename = self._get_kitti_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_indexes):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _get_kitti_results_file_template(self):
        filename = 'KITTI_' + '{:s}.txt'
        filedir = os.path.join(self.root_path, 'results', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _do_python_eval(self, output_dir='output'):
        cachedir = os.path.join(self.root_path, 'annotations_cache')
        aps = []
        class_nums = []

        use_07_metric = True
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self._get_kitti_results_file_template().format(cls)
            rec, prec, ap, gt_num = kitti_eval(filename, self.img_path, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            class_nums += [gt_num]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for i, ap in enumerate(aps):
            print(('{}: %{:.3f} out of {}'.format(self.classes[i], ap, class_nums[i])))
        print(('Average: {:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')



class DataLoader:
    def __init__(self, kitti, batch_size, shuffle=True):
        self.kitti = kitti
        self.batch_size = batch_size
        self.idx = list(range(kitti.size))
        self.shuffle = shuffle

    def get_batch(self):
        if self.shuffle:
            shuffle(self.idx)

        size = 0
        gt_boxes = []
        gt_classes = []
        imgs = []
        dont_care = []
        ori_ims = []
        for i in self.idx:
            if size == 0:
                gt_boxes = []
                gt_classes = []
                imgs = []
                dont_care = []
                ori_ims = []

            im, gt_box, gt_class, ori_im = self.kitti[i]
            gt_boxes.append(gt_box)
            gt_classes.append(gt_class)
            imgs.append(im)
            dont_care.append([])
            ori_ims.append(ori_im)


            size += 1
            if size == self.batch_size:
                size = 0
                imgs = torch.stack(imgs)
                yield imgs, gt_boxes, gt_classes, dont_care, ori_ims

