'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'LQ_condition':
        from data.LQ_condition_dataset import LQ_Dataset as D
    elif mode == 'LQGT_condition':
        from data.LQGT_condition_dataset import LQGT_dataset as D
    elif mode == 'LDRsToHDR':
        from data.LDRsToHDR_dataset import LDRsToHDR_dataset as D
    elif mode == 'LDRs':
        from data.LDRs_dataset import LDRs_dataset as D
    elif mode == '6LDRsToHDR':
        from data.LDRsToHDR_data import LDRsToHDR_dataset as D
    elif mode == '6LDRs':
        from data.LDRs_data import LDRs_dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
