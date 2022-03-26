import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import random

class LDRs_dataset(data.Dataset):
    '''Read LQ images only in the test phase.'''

    def __init__(self, opt):
        super(LDRs_dataset, self).__init__()
        self.opt = opt
        self.paths_LDRs = None
        self.LDRs_env = None  # environment for lmdb
        self.data_type = opt['data_type']
        # read image list from lmdb or image files
        print('*******************')
        print(opt['dataroot_LDRs'])
        self.sizes_ldr, self.paths_ldr = util.get_image_paths(self.data_type, opt['dataroot_LDRs'])
        # print(self.paths_ldr)

        self.paths_short_ldr = util.get_paths(opt['dataroot_LDRs'], '*_short.png')
        self.paths_medium_ldr = util.get_paths(opt['dataroot_LDRs'], '*_medium.png')
        self.paths_long_ldr = util.get_paths(opt['dataroot_LDRs'], '*_long.png')
        self.paths_exposures = util.get_paths(opt['dataroot_LDRs'], '*_exposures.npy')

        assert self.paths_short_ldr, 'Error: LDRs paths are empty.'

    def __getitem__(self, index):
        short_ldr_path = None
    
        # get exposures
        exposures = np.load(self.paths_exposures[index])
        floating_exposures = exposures - exposures[1]

        # get LDRs image
        ldr_images = []
        short_ldr_paths = self.paths_short_ldr[index]
        short_ldr_images = util.read_imgdata(short_ldr_paths, ratio=255.0)

        medium_ldr_paths = self.paths_medium_ldr[index]
        medium_ldr_images = util.read_imgdata(medium_ldr_paths, ratio=255.0)

        long_ldr_paths = self.paths_long_ldr[index]
        long_ldr_images = util.read_imgdata(long_ldr_paths, ratio=255.0)

        H, W, C = short_ldr_images.shape

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

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)

        return {'Short': img0, 'Medium': img1, 'Long': img2,  'short_path': short_ldr_paths}
        
    def __len__(self):
        return len(self.paths_exposures)
