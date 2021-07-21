from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model import ReconstructionModel
from torch.optim.lr_scheduler import MultiStepLR
from sync_batchnorm import DataParallelWithCallback
from frames_dataset import DatasetRepeater
import os
from sync_batchnorm import SynchronizedBatchNorm2d


def save_latest_ckpts(model_dict, iter_num, log_dir, train_mode, epoch):
    cpk = {k: v.state_dict() for k, v in model_dict.items()}
    cpk['epoch_' + train_mode] = epoch
    basename = '{}.pth'.format(iter_num)
    cpk_path = os.path.join(log_dir, basename)
    torch.save(cpk, cpk_path)


def train_customer_data(config, generator, region_predictor, bg_predictor, checkpoint, log_dir, dataset, device_ids):


    #for param in region_predictor.parameters():
    #    param.requires_grad = False
    # for param in bg_predictor.parameters():
    #    param.requires_grad = False

    for n,p in generator.named_parameters():

        if 'pixelwise_flow_predictor' in n:
            p.requires_grad = False
            print (n)


    train_params = config['train_params']
    print (train_params['lr'])
    optimizer = torch.optim.Adam(
                                list(generator.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, region_predictor, bg_predictor, None,
                                      None, None)
    else:
        start_epoch = 0
    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    # train_params['num_repeats'] = 2
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
       dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    model = ReconstructionModel(region_predictor, bg_predictor, generator, train_params)

    print (model)
    print (len(dataset))

    if torch.cuda.is_available():
        if ('use_sync_bn' in train_params) and train_params['use_sync_bn']:
            model = DataParallelWithCallback(model, device_ids=device_ids)
        else:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

    for module in model.modules():
        if isinstance(module, SynchronizedBatchNorm2d):

            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    total_iters = 0
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq'], train_mode = 'rebecca_taylor') as logger:

        for epoch in trange(start_epoch, train_params['num_epochs']):

            for idx, x in enumerate(dataloader):

                losses, generated = model(x)
                loss_values = [val.mean() for val in losses.values()]
                loss = sum(loss_values)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses.items()}
                logger.log_iter(losses=losses)
                #
                total_iters += 1

            # scheduler.step()
            logger.log_epoch(epoch, {'generator': generator,
                                     'bg_predictor': bg_predictor,
                                     'region_predictor': region_predictor,
                                     'optimizer_reconstruction': optimizer}, inp=x, out=generated)
