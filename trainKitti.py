import os
import torch
import datetime
import numpy as np
from darknetKitti import Darknet19

from datasets.kitti import Kitti
from datasets.kitti import DataLoader
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.configK as cfg
from random import randint, shuffle

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

# data loader
epochs = 50
kitti = Kitti(cfg.DATA_DIR, cfg=cfg, stage='train')
train_loader = torch.utils.data.DataLoader(kitti, batch_size=cfg.train_batch_size, shuffle=True)

# dst_size=cfg.inp_size)
print('load data succ...')

net = Darknet19()
# net_utils.load_net(cfg.trained_model, net)
# pretrained_model = os.path.join(cfg.train_output_dir,
#     'darknet19_voc07trainval_exp1_63.h5')
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
net.load_from_npz(cfg.pretrained_model, num_conv=18)
net.cuda()
net.train()

print('load net succ...')

# optimizer
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

# tensorboad
use_tensorboard = cfg.use_tensorboard and SummaryWriter is not None
# use_tensorboard = False
if use_tensorboard:
    summary_writer = SummaryWriter(os.path.join(cfg.TRAIN_DIR, 'runs', cfg.exp_name))
else:
    summary_writer = None

batch_per_epoch = kitti.size // cfg.train_batch_size
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0
t = Timer()
step_cnt = 0
size_index = 0

dt_loader = DataLoader(kitti, cfg.train_batch_size)

step = 0
for epoch in range(epochs):
    for im_data, gt_boxes, gt_classes, dontcare, ori_imgs in dt_loader.get_batch():
        step += 1
        t.tic()
        # forward
        # im_data = net_utils.np_to_variable(im, is_cuda=True, volatile=False).permute(0, 3, 1, 2)
        bbox_pred, iou_pred, prob_pred = net(im_data.cuda(), gt_boxes, gt_classes, dontcare, size_index)

        # backward
        loss = net.loss
        bbox_loss += net.bbox_loss.data.cpu().item()
        iou_loss += net.iou_loss.data.cpu().item()
        cls_loss += net.cls_loss.data.cpu().item()
        train_loss += loss.data.cpu().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
        step_cnt += 1
        duration = t.toc()
        if step % cfg.disp_interval == 0:
            train_loss /= cnt
            bbox_loss /= cnt
            iou_loss /= cnt
            cls_loss /= cnt
            print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
                   'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
                   (epoch, step_cnt, batch_per_epoch, train_loss, bbox_loss,
                    iou_loss, cls_loss, duration,
                    str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))  # noqa

            if summary_writer and step % cfg.log_interval == 0:
                summary_writer.add_scalar('loss_train', train_loss, step)
                summary_writer.add_scalar('loss_bbox', bbox_loss, step)
                summary_writer.add_scalar('loss_iou', iou_loss, step)
                summary_writer.add_scalar('loss_cls', cls_loss, step)
                summary_writer.add_scalar('learning_rate', lr, step)

                # plot results
                bbox_pred = bbox_pred.data[0:1].cpu().numpy()
                iou_pred = iou_pred.data[0:1].cpu().numpy()
                prob_pred = prob_pred.data[0:1].cpu().numpy()
                # image = im_data.cpu().data.numpy()[0]
                # image = np.rollaxis(image, 0, 3)
                image = ori_imgs[0]
                # bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, (image.shape[1], image.shape[0]), cfg, thresh=0.3, size_index=size_index)
                bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh=0.3, size_index=size_index)

                im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)
                summary_writer.add_image('predict', im2show, step)

                im2show = yolo_utils.draw_detection(image, np.round(gt_boxes[0]).astype(int),
                                                    np.array([1] * gt_boxes[0].shape[0]), gt_classes[0], cfg, thr=0.1)
                summary_writer.add_image('GT', im2show, step)

            train_loss = 0
            bbox_loss, iou_loss, cls_loss = 0., 0., 0.
            cnt = 0
            t.clear()
            size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
            print("image_size {}".format(cfg.multi_scale_inp_size[size_index]))

        if step > 0 and (step % batch_per_epoch == 0):
            if epoch in cfg.lr_decay_epochs:
                lr *= cfg.lr_decay
                optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                            momentum=cfg.momentum,
                                            weight_decay=cfg.weight_decay)

            save_name = os.path.join(cfg.train_output_dir,
                                     '{}_{}.h5'.format(cfg.exp_name, epoch))
            net_utils.save_net(save_name, net)
            print(('save model: {}'.format(save_name)))
            step_cnt = 0
