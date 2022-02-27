
Online



Add a channel description



At 6:17 PM Saturday, February 26, isabella wrote, python test_psmnet_with_confidence.py --config-file configs/remote_test.yaml --model ../train_8_14_cascade/train1/models/model_best.pth --onreal --exclude-bg --exclude-zeros
1 new message today
	
isabella
12:42 PM
我是指的紫色的mask

	
jianyu
Update your status
12:42 PM
紫色的不是0吗

	
isabella
12:43 PM
对 在prediction 里面 对应的紫色区域mask掉

我一会给的new pred depth

	
jianyu
Update your status
12:43 PM
mask过了

	
isabella
12:43 PM
??

zoom一下？

	
jianyu
Update your status
12:44 PM
行

	
isabella
12:44 PM
 https://ucsd.zoom.us/j/95952636223

Zoom Video
Join our Cloud HD Video Meeting
Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom Rooms is the original software-based conference room solution used around the world in board, con...
	
jianyu
Update your status
1:47 PM
video_3_pred.mp4
MP42.6MB
video_3_realsense.mp4
MP42.8MB
	
isabella
9:57 PM
https://drive.google.com/file/d/1HM8DSDvDvhJKBO5oLWwNJp1wxkLD0BU2/view?usp=sharing

在吗 

试下这个video3

zoom now?

	
jianyu
Update your status
9:58 PM
ok

	
isabella
10:23 PM
0191.png


0000.png


	
isabella
10:32 PM
0003.png


	
jianyu
Update your status
11:39 PM
p.py
PY6KB
February 03
	
jianyu
Update your status
3:52 PM
你觉得我们应该zoom in哪些地方

	
isabella
3:58 PM
你觉得？

	
jianyu
Update your status
3:58 PM
15s我们的model好像fill in了realsense的缺口

...

这是我能找出来比较明显的了

	
isabella
3:58 PM
我感觉现在搞有点晚了...rebuttal已经快完了

	
jianyu
Update your status
3:59 PM
我不知道edward有没有传

	
isabella
4:00 PM
:joy: 

	
jianyu
Update your status
4:23 PM
他没有传

那现在还能传么

万一有reviewer看到了有reviewer没看到怎么办

。。

	
isabella
4:27 PM
那你还是先做着吧。。

今晚加快搞个完整版 发群里再问一下Hao看看要不要发

关键他不回复。。就很迷

	
jianyu
Update your status
4:30 PM
true

	
isabella
4:30 PM
要不你再问问他

Yesterday
	
jianyu
Update your status
9:39 PM
那个根据deptherr来定颜色的pc code push了吗

	
isabella
9:45 PM
test_util.py
PY8KB
似乎没有...

你用这个吧

line 111-116

np_utils.py
PY941B
replace原来的test_util就行了

	
jianyu
Update your status
10:05 PM
ok

谢啦

Today
	
jianyu
Update your status
4:17 PM
那个test script能给我一份不

	
isabella
5:47 PM
test_psmnet_with_confidence.py
PY14KB
psmnet_confidence.py
PY9KB
	
jianyu
Update your status
5:50 PM
hao

"pip install --user open3d && sh /jianyu-fast-vol/ActiveZero/run_test.sh"

这样吗

	
isabella
5:52 PM
en

暂时可以这样 之后我还是要更新一下docker

	
jianyu
Update your status
5:55 PM
item['intrinsic_l'] = torch.tensor(intrinsic_l)

messytable_test

里是不是要加这一行

	
isabella
5:58 PM
messytable_test.py
PY12KB
	
jianyu
Update your status
5:59 PM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: -9) local_rank: 0 (pid: 141) of binary: /opt/conda/bin/python
ERROR:torch.distributed.elastic.agent.server.local_elastic_agent:[default] Worker group failed
。。。

这是咋回事

	
isabella
6:01 PM
不知道...

	
jianyu
Update your status
6:12 PM
你是怎么run的

distributed.launch?

要不你吧整个重新发我一下吧

我还是修不好

	
isabella
6:16 PM
test 不用distributed啊

New Messages
python test_psmnet_with_confidence.py --config-file configs/remote_test.yaml --model ../train_8_14_cascade/train1/models/model_best.pth --onreal --exclude-bg --exclude-zeros

Write to isabella

No file chosen

isabella is typing...
Help
"""
Author: Isabella Liu 8/15/21
Feature:
"""

import os
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms as Transforms
from torch.utils.data import Dataset, DataLoader
import cv2

from utils.config import cfg
from utils.util import load_pickle
from utils.test_util import calc_left_ir_depth_from_rgb


def gamma_trans(img, gamma):
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def __data_augmentation__(gaussian_blur=False, color_jitter=False):
    """
    :param gaussian_blur: Whether apply gaussian blur in data augmentation
    :param color_jitter: Whether apply color jitter in data augmentation
    Note:
        If you want to change the parameters of each augmentation, you need to go to config files,
        e.g. configs/remote_train_config.yaml
    """
    transform_list = [
        Transforms.ToTensor()
    ]
    if gaussian_blur:
        gaussian_sig = random.uniform(cfg.DATA_AUG.GAUSSIAN_MIN, cfg.DATA_AUG.GAUSSIAN_MAX)
        transform_list += [
            Transforms.GaussianBlur(kernel_size=cfg.DATA_AUG.GAUSSIAN_KERNEL, sigma=gaussian_sig)
        ]
    if color_jitter:
        bright = random.uniform(cfg.DATA_AUG.BRIGHT_MIN, cfg.DATA_AUG.BRIGHT_MAX)
        contrast = random.uniform(cfg.DATA_AUG.CONTRAST_MIN, cfg.DATA_AUG.CONTRAST_MAX)
        transform_list += [
            Transforms.ColorJitter(brightness=[bright, bright],
                                   contrast=[contrast, contrast])
        ]
    # Normalization
    transform_list += [
        Transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ]
    custom_augmentation = Transforms.Compose(transform_list)
    return custom_augmentation


class MessytableTestDataset(Dataset):
    def __init__(self, split_file, debug=False, sub=100, onReal=False):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        """
        self.img_L, self.img_R, self.img_L_real, self.img_R_real, self.img_depth_l, self.img_depth_r, \
        self.img_meta, self.img_label, self.img_sim_realsense, self.img_real_realsense, self.mask_scenes \
            = self.__get_split_files__(split_file, debug=debug, sub=sub)
        self.onReal = onReal
        self.brightness_factor = 1.8

    @staticmethod
    def __get_split_files__(split_file, debug=False, sub=100):
        """
        :param split_file: Path to the split .txt file, e.g. train.txt
        :param debug: Debug mode, load less data
        :param sub: If debug mode is enabled, sub will be the number of data loaded
        :return: Lists of paths to the entries listed in split file
        """
        sim_dataset = cfg.DIR.DATASET
        real_dataset = cfg.REAL.DATASET
        sim_img_left_name = cfg.SPLIT.LEFT
        sim_img_right_name = cfg.SPLIT.RIGHT
        real_img_left_name = cfg.REAL.LEFT
        real_img_right_name = cfg.REAL.RIGHT
        sim_realsense = cfg.SPLIT.SIM_REALSENSE
        real_realsense = cfg.SPLIT.REAL_REALSENSE

        with open(split_file, 'r') as f:
            prefix = [line.strip() for line in f]

            img_L_sim = [os.path.join(sim_dataset, p, sim_img_left_name) for p in prefix]
            img_R_sim = [os.path.join(sim_dataset, p, sim_img_right_name) for p in prefix]
            img_L_real = [os.path.join(real_dataset, p, real_img_left_name) for p in prefix]
            img_R_real = [os.path.join(real_dataset, p, real_img_right_name) for p in prefix]
            img_depth_l = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTHL) for p in prefix]
            img_depth_r = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.DEPTHR) for p in prefix]
            img_meta = [os.path.join(cfg.DIR.DATASET, p, cfg.SPLIT.META) for p in prefix]
            img_label = [os.path.join(cfg.REAL.DATASET_V9, p, cfg.SPLIT.LABEL) for p in prefix]
            img_sim_realsense = [os.path.join(sim_dataset, p, sim_realsense) for p in prefix]
            img_real_realsense = [os.path.join(real_dataset, p, real_realsense) for p in prefix]

            if debug is True:
                img_L_sim = img_L_sim[:sub]
                img_R_sim = img_R_sim[:sub]
                img_L_real = img_L_real[:sub]
                img_R_real = img_R_real[:sub]
                img_depth_l = img_depth_l[:sub]
                img_depth_r = img_depth_r[:sub]
                img_meta = img_meta[:sub]
                img_label = img_label[:sub]
                img_sim_realsense = img_sim_realsense[:sub]
                img_real_realsense = img_real_realsense[:sub]

        # Obtain robot arm mask list
        with open(cfg.REAL.MASK_FILE, 'r') as f:
            mask_scenes = [line.strip() for line in f]

        return img_L_sim, img_R_sim, img_L_real, img_R_real, img_depth_l, img_depth_r, img_meta, img_label, \
            img_sim_realsense, img_real_realsense, mask_scenes

    def __len__(self):
        return len(self.img_L)

    def __getitem__(self, idx):
        if self.onReal:
            # Adjust brightness of real images
            img_L_rgb = Image.open(self.img_L_real[idx]).convert(mode='L')
            img_R_rgb = Image.open(self.img_R_real[idx]).convert(mode='L')
            # img_L_rgb = gamma_trans(np.array(img_L_rgb), 0.5)
            # img_R_rgb = gamma_trans(np.array(img_R_rgb), 0.5)
            img_L_rgb = np.array(img_L_rgb)
            img_R_rgb = np.array(img_R_rgb)
            img_L_rgb = img_L_rgb[:, :, None]
            img_R_rgb = img_R_rgb[:, :, None]
            img_L_rgb = np.repeat(img_L_rgb, 3, axis=-1)
            img_R_rgb = np.repeat(img_R_rgb, 3, axis=-1)

            img_L_rgb_sim = np.array(Image.open(self.img_L[idx]).convert(mode='L')) / 255
            img_R_rgb_sim = np.array(Image.open(self.img_R[idx]).convert(mode='L')) / 255
            img_L_rgb_sim = np.repeat(img_L_rgb_sim[:, :, None], 3, axis=-1)
            img_R_rgb_sim = np.repeat(img_R_rgb_sim[:, :, None], 3, axis=-1)
            img_depth_realsense = np.array(Image.open(self.img_real_realsense[idx])) / 1000

        else:
            img_L_rgb = np.array(Image.open(self.img_L[idx]).convert(mode='L')) / 255
            img_R_rgb = np.array(Image.open(self.img_R[idx]).convert(mode='L')) / 255
            img_L_rgb = np.repeat(img_L_rgb[:, :, None], 3, axis=-1)
            img_R_rgb = np.repeat(img_R_rgb[:, :, None], 3, axis=-1)
            img_L_rgb_real = np.array(Image.open(self.img_L_real[idx]).convert(mode='L'))[:, :, None]
            img_R_rgb_real = np.array(Image.open(self.img_R_real[idx]).convert(mode='L'))[:, :, None]
            img_L_rgb_real = np.repeat(img_L_rgb_real, 3, axis=-1)
            img_R_rgb_real = np.repeat(img_R_rgb_real, 3, axis=-1)
            img_depth_realsense = np.array(Image.open(self.img_sim_realsense[idx])) / 1000

        img_depth_l = np.array(Image.open(self.img_depth_l[idx])) / 1000  # convert from mm to m
        img_depth_r = np.array(Image.open(self.img_depth_r[idx])) / 1000  # convert from mm to m
        img_meta = load_pickle(self.img_meta[idx])
        img_label = np.array(Image.open(self.img_label[idx]))

        # Convert depth map to disparity map
        extrinsic = img_meta['extrinsic']
        extrinsic_l = img_meta['extrinsic_l']
        extrinsic_r = img_meta['extrinsic_r']
        intrinsic = img_meta['intrinsic']
        intrinsic_l = img_meta['intrinsic_l']
        baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
        focal_length = intrinsic_l[0, 0] / 2

        mask = img_depth_l > 0
        img_disp_l = np.zeros_like(img_depth_l)
        img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]
        mask = img_depth_r > 0
        img_disp_r = np.zeros_like(img_depth_r)
        img_disp_r[mask] = focal_length * baseline / img_depth_r[mask]

        # Mask out the robot arm, mask images are stored in 1280 * 720
        prefix = self.img_L[idx].split('/')[-2]
        scene_id = prefix.split('-')[-1]
        if scene_id in self.mask_scenes:
            robot_mask_file = os.path.join(cfg.REAL.MASK, scene_id + '.png')
            robot_mask = Image.open(robot_mask_file).convert(mode='L')
            h, w = mask.shape
            robot_mask = robot_mask.resize((w,h), resample=Image.BILINEAR)
            robot_mask = np.array(robot_mask) / 255
        else:
            robot_mask = np.zeros_like(img_depth_l)

        # Convert img_depth_realsense to irL camera frame
        img_depth_realsense = calc_left_ir_depth_from_rgb(intrinsic, intrinsic_l,
                                                          extrinsic, extrinsic_l, img_depth_realsense)

        # Get data augmentation
        # custom_augmentation = __data_augmentation__(gaussian_blur=self.gaussian_blur, color_jitter=self.color_jitter)
        normalization = __data_augmentation__(gaussian_blur=False, color_jitter=False)

        item = {}
        item['img_L'] = normalization(img_L_rgb).type(torch.FloatTensor)
        item['img_R'] = normalization(img_R_rgb).type(torch.FloatTensor)
        item['img_disp_l'] = torch.tensor(img_disp_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_l'] = torch.tensor(img_depth_l, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_disp_r'] = torch.tensor(img_disp_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_r'] = torch.tensor(img_depth_r, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_depth_realsense'] = torch.tensor(img_depth_realsense, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['img_label'] = torch.tensor(img_label, dtype=torch.float32).unsqueeze(0)  # [bs, 1, H, W]
        item['prefix'] = prefix
        item['focal_length'] = torch.tensor(focal_length, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['baseline'] = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item['robot_mask'] = torch.tensor(robot_mask, dtype=torch.float32).unsqueeze(0)
        item["intrinsic_l"] = torch.tensor(intrinsic_l, dtype=torch.float32)

        if self.onReal is False:
            item['img_L_real'] = normalization(img_L_rgb_real).type(torch.FloatTensor)
            item['img_R_real'] = normalization(img_R_rgb_real).type(torch.FloatTensor)
        else:
            item['img_L_sim'] = normalization(img_L_rgb_sim).type(torch.FloatTensor)
            item['img_R_sim'] = normalization(img_R_rgb_sim).type(torch.FloatTensor)

        return item


def get_test_loader(split_file, debug=False, sub=100, onReal=False):
    """
    :param split_file: split file
    :param debug: Whether on debug mode, load less data
    :param sub: If on debug mode, how many items to load into dataset
    :param isTest: Whether on test, if test no random crop on input image
    :param onReal: Whether test on real dataset, folder and file name are different
    :return: dataloader
    """
    messytable_dataset = MessytableTestDataset(split_file, debug, sub, onReal=onReal)
    loader = DataLoader(messytable_dataset, batch_size=1, num_workers=0)
    return loader


if __name__ == '__main__':
    cdataset = MessytableTestDataset('/code/dataset_local_v9/training_lists/all.txt', onReal=True)
    item = cdataset.__getitem__(0)
    print(item['img_L'].shape)
    print(item['img_R'].shape)
    # print(item['img_L_real'].shape)
    print(item['img_disp_l'].shape)
    print(item['prefix'])
    print(item['img_depth_realsense'].shape)
    print(item['robot_mask'].shape)