import os
import cv2
import numpy as np
import pickle
import argparse

from datasets.kitti import Kitti
from datasets.kitti import DataLoader
from darknetKitti import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from datasets.pascal_voc import VOCDataset
import cfgs.configK as cfg


parser = argparse.ArgumentParser(description='PyTorch Yolo')
parser.add_argument('--image_size_index', type=int, default=0,
                    metavar='image_size_index',
                    help='setting images size index 0:320, 1:352, 2:384, 3:416, 4:448, 5:480, 6:512, 7:544, 8:576')
args = parser.parse_args()


# hyper-parameters
# ------------
imdb_name = cfg.imdb_test
# trained_model = cfg.trained_model
trained_model = os.path.join(cfg.train_output_dir, 'darknet19_kittitrainval_exp3_49.h5')
output_dir = cfg.test_output_dir

max_per_image = 300
thresh = 0.01
vis = False
# ------------


def test_net(net, imdb, max_per_image=300, thresh=0.5, vis=False, num_classes=8):
    num_images = imdb.size

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    dt_loader = DataLoader(kitti, batch_size=1, shuffle=False)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    size_index = args.image_size_index
    i = 0
    for im_data, gt_boxes, gt_classes, dontcare, ori_imgs in dt_loader.get_batch():

        # ori_im = (np.rollaxis(im_data[0].data.numpy(), 0, 3) * 255).astype(np.uint8)
        ori_im = ori_imgs[0]

        _t['im_detect'].tic()
        bbox_pred, iou_pred, prob_pred = net(im_data.cuda())

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        # bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, (ori_im.shape[1], ori_im.shape[0]), cfg, thresh, size_index)
        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, ori_im.shape, cfg, thresh, size_index)
        detect_time = _t['im_detect'].toc()

        _t['misc'].tic()

        for j in range(num_classes):
            inds = np.where(cls_inds == j)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_dets = np.hstack((c_bboxes,
                                c_scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = c_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))  # noqa
            _t['im_detect'].clear()
            _t['misc'].clear()

        if vis:
            im2show = yolo_utils.draw_detection(ori_im, bboxes, scores, cls_inds, cfg, thr=0.1)
            # im2show = yolo_utils.draw_detection(ori_im, np.round(gt_boxes[0]).astype(int), np.array([1]*gt_boxes[0].shape[0]), gt_classes[0], cfg, thr=0.1)

            if im2show.shape[0] > 1100:
                im2show = cv2.resize(im2show, (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))  # noqa
            print im2show.shape
            cv2.imshow('test', im2show)
            cv2.waitKey(0)
        i += 1

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    kitti.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    # data loader
    kitti = Kitti(cfg.DATA_DIR, cfg=cfg, stage='val')

    net = Darknet19()
    net_utils.load_net(trained_model, net)

    net.cuda()
    net.eval()

    test_net(net, kitti, max_per_image, thresh, vis, cfg.num_classes)

