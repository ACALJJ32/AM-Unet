import torch
import logging

import models.modules.UNet_arch as UNet_arch
import models.modules.ADNet_model as ADNet_model

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'HDRUNet':
        netG = UNet_arch.HDRUNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], act_type=opt_net['act_type'])
    elif which_model == 'ADNet':
        netG = ADNet_model.ADNet(6, 5, 64, 32)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG