import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os.path as osp


class LDRsToHDR_dataset(data.Dataset):

    def __init__(self, opt):
        super(LDRsToHDR_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None

        self.sizes_short_ldr, self.paths_short_ldr = util.get_image_paths(self.data_type, opt['dataroot_short'])
        self.sizes_medium_ldr, self.paths_medium_ldr = util.get_image_paths(self.data_type, opt['dataroot_medium'])
        self.sizes_long_ldr, self.paths_long_ldr = util.get_image_paths(self.data_type, opt['dataroot_long'])

        self.paths_exposures = opt['dataroot_exp']

        self.sizes_GT, self.paths_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.folder_ratio = opt['dataroot_ratio']


    def __getitem__(self, index):
        GT_path = None 
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get LDR images
        ldr_images = []
        short_ldr_paths = self.paths_short_ldr[index]
        short_ldr_images = util.read_imgdata(short_ldr_paths, ratio=255.0)

        medium_ldr_paths = self.paths_medium_ldr[index]
        medium_ldr_images = util.read_imgdata(medium_ldr_paths, ratio=255.0)

        long_ldr_paths = self.paths_long_ldr[index]
        long_ldr_images = util.read_imgdata(long_ldr_paths, ratio=255.0)

        # get GT alignratio
        filename = osp.basename(short_ldr_paths)[:4] + "_alignratio.npy"
        ratio_path = osp.join(self.folder_ratio, filename)
        alignratio = np.load(ratio_path).astype(np.float32)

        # get exposures
        exp_name = osp.basename(short_ldr_paths)[:4] + "_exposures.npy"
        exp_path = osp.join(self.paths_exposures, exp_name)
        exposures = np.load(exp_path)
        floating_exposures = exposures - exposures[1]

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = util.read_imgdata(GT_path, ratio=alignratio)

        if self.opt['phase'] == 'train':
            
            H, W, C = short_ldr_images.shape
            H_gt, W_gt, C = img_GT.shape
            if H != H_gt:
                print('*******wrong image*******:{}'.format(index))
            LQ_size = GT_size // scale
            # LQ_size = GT_size
            
            
            # randomly crop
            if GT_size != 0:
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                long_ldr_images = long_ldr_images[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] #256*256*3
                medium_ldr_images = medium_ldr_images[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                short_ldr_images = short_ldr_images[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            short_ldr_images, medium_ldr_images, long_ldr_images, img_GT = util.augment([short_ldr_images, medium_ldr_images, long_ldr_images, img_GT], self.opt['use_flip'],
                                        self.opt['use_rot'])
        else:
            H, W, C = short_ldr_images.shape
            H_gt, W_gt, C = img_GT.shape
            if H != H_gt:
                print('*******wrong image*******:{}'.format(index))
            LQ_size = GT_size // scale
            # LQ_size = GT_size
            # randomly crop
            if GT_size != 0:
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                long_ldr_images = long_ldr_images[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] #256*256*3
                medium_ldr_images = medium_ldr_images[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                short_ldr_images = short_ldr_images[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            short_ldr_images, medium_ldr_images, long_ldr_images, img_GT = util.augment([short_ldr_images, medium_ldr_images, long_ldr_images, img_GT], self.opt['use_flip'],
                                        self.opt['use_rot'])

        ldr_images.append(short_ldr_images)
        ldr_images.append(medium_ldr_images)
        ldr_images.append(long_ldr_images)
        ldr_images = np.array(ldr_images)
        # ldr images process
        s_gamma = 2.24
        if random.random() < 0.3:
            s_gamma += (random.random() * 0.2 - 0.1)
        image_short = util.ev_alignment(ldr_images[0], floating_exposures[0], s_gamma)
        # image_medium = ev_alignment(ldr_images[1], floating_exposures[1], 2.24)
        image_medium = ldr_images[1]
        image_long = util.ev_alignment(ldr_images[2], floating_exposures[2], s_gamma)

        image_short_concat = np.concatenate((image_short, short_ldr_images), 2)  
        image_medium_concat = np.concatenate((image_medium, medium_ldr_images), 2)
        image_long_concat = np.concatenate((image_long, long_ldr_images), 2)

        img0 = image_short_concat.astype(np.float32).transpose(2, 0, 1)
        img1 = image_medium_concat.astype(np.float32).transpose(2, 0, 1)
        img2 = image_long_concat.astype(np.float32).transpose(2, 0, 1)
        img_GT = img_GT.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0) # [6,256,256]
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        img_GT = torch.from_numpy(img_GT)
        # print(img_GT.shape) # (3, 160, 160)
        sample = {'Short': img0, 'Medium': img1, 'Long': img2, 'GT': img_GT, 'GT_path': GT_path}
        # print(sample)
        return sample

    def __len__(self):
        return len(self.paths_GT)

