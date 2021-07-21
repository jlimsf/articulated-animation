from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model import ReconstructionModel
from torch.optim.lr_scheduler import MultiStepLR
from sync_batchnorm import DataParallelWithCallback
from frames_dataset import DatasetRepeater
import os
import numpy as np
from sync_batchnorm import SynchronizedBatchNorm2d

def save_latest_ckpts(model_dict, iter_num, log_dir, train_mode, epoch):
    cpk = {k: v.state_dict() for k, v in model_dict.items()}
    cpk['epoch_' + train_mode] = epoch
    basename = '{}.pth'.format(iter_num)
    cpk_path = os.path.join(log_dir, basename)
    torch.save(cpk, cpk_path)


def train_gan(config, generator, region_predictor, bg_predictor, checkpoint, log_dir, dataset, device_ids, discriminator, avd_network, finetune=False):


    #for param in region_predictor.parameters():
    #    param.requires_grad = False
    # for param in bg_predictor.parameters():
    #    param.requires_grad = False

    train_params = config['train_params']

    optimizer = torch.optim.Adam(list(generator.parameters()) +
                                 list(region_predictor.parameters()) +
                                 list(bg_predictor.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr'], betas=(0.9, 0.999))
    gan_criterion = torch.nn.BCEWithLogitsLoss()
    mse_criterion = torch.nn.MSELoss()

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, region_predictor, bg_predictor, avd_network,
                                      None, None)
    else:
        start_epoch = 0
    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)


    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
       dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    print (len(dataset))

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    model = ReconstructionModel(region_predictor, bg_predictor, generator, train_params)

    if torch.cuda.is_available():
        if ('use_sync_bn' in train_params) and train_params['use_sync_bn']:
            model = DataParallelWithCallback(model, device_ids=device_ids)
        else:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

    if finetune:
        print ("Freezing parameters")
        for n,p in generator.named_parameters():

            if 'pixelwise_flow_predictor' in n:
                p.requires_grad = False
                print (n)

        for module in model.modules():
            if isinstance(module, SynchronizedBatchNorm2d):

                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    total_iters = 0


    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq'], train_mode = 'customer_data') as logger:

        for epoch in trange(start_epoch, train_params['num_epochs']):

            for param_group in optimizer.param_groups:
                print(param_group['lr'])

            for idx, x in enumerate(dataloader):

                optimizer.zero_grad()
                losses, generated = model(x)

                # Foold discriminator
                fake_preds = discriminator(generated['prediction'] )

                real_labels = torch.ones_like(fake_preds).cuda()


                # a = list(model.module.generator.pixelwise_flow_predictor.parameters())[0].clone()
                Gloss = gan_criterion(fake_preds, real_labels) * 10.

                loss_values = [val.mean() for val in losses.values()]
                loss = sum(loss_values)
                # print (loss)
                loss += Gloss
                # print (Gloss)
                loss.backward()
                optimizer.step()

                # b = list(model.module.generator.pixelwise_flow_predictor.parameters())[0].clone()
                # print(torch.equal(a.data, b.data))

                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses.items()}
                logger.log_iter(losses=losses)
                #
                total_iters += 1

                optimizer_D.zero_grad()

                fake_preds = discriminator(generated['prediction'].detach())
                real_preds = discriminator(x['source'].cuda())

                fake_d_loss = gan_criterion(fake_preds, torch.zeros_like(fake_preds).cuda())
                real_d_loss = gan_criterion(real_preds, torch.ones_like(real_preds).cuda())

                Dloss = fake_d_loss + real_d_loss
                Dloss *= 5.


                Dloss.backward()
                optimizer_D.step()
                # print (losses['perceptual'], Dloss.item(), Gloss.item())
                # print (loss)
                # exit()

            # scheduler.step()
            logger.log_epoch(epoch, {'generator': generator,
                                     'bg_predictor': bg_predictor,
                                     'region_predictor': region_predictor,
                                     'optimizer_reconstruction': optimizer}, inp=x, out=generated)
