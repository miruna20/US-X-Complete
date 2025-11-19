import subprocess

import numpy as np
import torch.optim as optim
import torch
# from utils.train_utils import *
import wandb
from train_utils import *
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
from dataset import MVP_CP
from dataset import verse2020_lumbar
import os
from torch.nn.parallel import DistributedDataParallel

# optional wandb import
try:
    import wandb
except ImportError:
    wandb = None

import warnings

warnings.filterwarnings("ignore")


device_ids = [0]
device = 'cuda'

wandb_enabled = False
wandb_run = None

def train():
    logging.info(str(args))
    if args.eval_emd:
        metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    else:
        metrics = ['cd_p', 'cd_t', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    use_labelmaps = (args.use_labelmaps_in_PMNET or args.use_labelmaps_in_RENet)

    dataset = verse2020_lumbar(train_path=args.path_to_train_dataset,
                               val_path=args.path_to_val_dataset,
                               test_path=args.path_to_test_dataset,
                               apply_trafo=args.apply_trafo,
                               sigma = args.sigma,
                               Xray_labelmap=use_labelmaps,
                               prefix = "train",
                               num_partial_scans_per_mesh=args.num_partial_scans_per_mesh,
                               )
    dataset_test = verse2020_lumbar(train_path=args.path_to_train_dataset,
                                    val_path=args.path_to_val_dataset,
                                    test_path=args.path_to_test_dataset,
                                    apply_trafo=args.apply_trafo,
                                    sigma=args.sigma,
                                    Xray_labelmap=use_labelmaps,
                                    prefix="val",
                                    num_partial_scans_per_mesh=args.num_partial_scans_per_mesh,
                                    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                            shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')

    net = torch.nn.DataParallel(model_module.Model(args))
    #net = torch.nn.DistributedDataParallel(model_module.Model(args),num_procs=1)
    #net = model_module.Model(args)
    net.to(device)
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)

    cascade_gan = (args.model_name == 'cascade')
    net_d = None
    if cascade_gan:
        net_d = torch.nn.DataParallel(model_module.Discriminator(args))
        net_d.to(device)
        net_d.module.apply(model_module.weights_init)

    lr = args.lr
    if cascade_gan:
        lr_d = lr / 2
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    if cascade_gan:
        optimizer_d = optim.Adam(net_d.parameters(), lr=lr_d, weight_decay=0.00001, betas=(0.5, 0.999))

    alpha = None
    if args.varying_constant:
        varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
        varying_constant = [float(c.strip()) for c in args.varying_constant.split(',')]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        if cascade_gan:
            net_d.module.load_state_dict(ckpt['D_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    for epoch in range(args.start_epoch, args.nepoch):
        train_loss_meter.reset()
        net.module.train()
        if cascade_gan:
            net_d.module.train()

        if args.varying_constant:
            for ind, ep in enumerate(varying_constant_epochs):
                if epoch < ep:
                    alpha = varying_constant[ind]
                    break
                elif ind == len(varying_constant_epochs)-1 and epoch >= ep:
                    alpha = varying_constant[ind+1]
                    break

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            if cascade_gan:
                optimizer_d.zero_grad()

            _, partial_pcd, labelmap, gt = data

            if(use_labelmaps):
                labelmap = labelmap.float().to(device)
                labelmap = labelmap.transpose(2, 1).contiguous()

            # mean_feature = None

            inputs = partial_pcd[0, :, :].cpu().numpy()
            gt_pcd = gt[0, :, :].cpu().numpy()
            filename = os.path.join('training_epoch_{:03d}.png'.format(epoch))
            if (use_labelmaps):
                labelmap_pcd = labelmap[0,:,:].cpu().numpy().T
                plot_pcd_one_view(filename=filename,
                                  pcds=[inputs, labelmap_pcd,gt_pcd],
                                  titles=["input_pcd","XrayLabelmap", "gt"])
            else:
                plot_pcd_one_view(filename=filename,
                                  pcds=[inputs, gt_pcd],
                                  titles=["input_pcd", "gt"])
            if wandb_enabled and wandb_run is not None:
                wandb.log({"inputs_sanitycheck_at_training_time": wandb.Image(filename)})



            partial_pcd = partial_pcd.float().to(device)
            partial_pcd = partial_pcd.transpose(2, 1).contiguous()

            gt = gt.float().to(device)

            # out2, loss2, net_loss = net(inputs, gt, mean_feature=mean_feature, alpha=alpha)
            out2, loss2, net_loss = net(partial_pcd, labelmap, gt, alpha=alpha)

            #if(epoch%10 == 0):
            # TODO add to the figure and only log it in the end of the epoch, otherwise we log per step
            inputs = partial_pcd[0, :, :].cpu().numpy().T
            fine_pcd = out2[0, :, :].detach().cpu().numpy()
            gt_pcd = gt[0, :, :].cpu().numpy()
            filename = os.path.join('training_epoch_{:03d}.png'.format(epoch))
            if(use_labelmaps):
                labelmap_pcd = labelmap[0, :, :].cpu().numpy().T
                plot_pcd_one_view(filename=filename,
                              pcds=[inputs, labelmap_pcd, fine_pcd, gt_pcd],
                              titles=["input_pcd", "XrayLabelmap", "fine", "gt"])
            else:
                plot_pcd_one_view(filename=filename,
                                  pcds=[inputs, fine_pcd, gt_pcd],
                                  titles=["input_pcd", "fine", "gt"])

            if wandb_enabled and wandb_run is not None:
                wandb.log({"combined_pcds_training": wandb.Image(filename)})

            if cascade_gan:
                d_fake = generator_step(net_d, out2, net_loss, optimizer)
                discriminator_step(net_d, gt, d_fake, optimizer_d)
            else:
                train_loss_meter.update(net_loss.mean().item())
                net_loss.backward(torch.squeeze(torch.ones(1).to(device)))
                optimizer.step()

            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, fine_loss: %f total_loss: %f lr: %f' %
                             (epoch, i, len(dataset) / args.batch_size, args.loss, loss2.mean().item(), net_loss.mean().item(), lr) + ' alpha: ' + str(alpha))
                if wandb_enabled and wandb_run is not None:
                    wandb.log({"fine loss":loss2.mean().item(),"net_loss":net_loss.mean().item()})
        if epoch % args.epoch_interval_to_save == 0:
            model_path = '%s/network.pth' % log_dir
            save_model(model_path, net, net_d=net_d)
            logging.info("Saving net...")

        # get a dictionary
        param_dict = dict(net.named_parameters())
        params_to_log = ["module.feature_selector.fc1.weight",
                         "module.feature_selector.fc2.weight"]

        # access only the feature selector weights
        half_size = 1024
        for param in params_to_log:
            # output them in a couple of ways in which we can monitor them
            layer_weights = param_dict[param].data.cpu().numpy()

            layer_weights_partialPCD = layer_weights[:,:half_size]
            layer_weights_partialPCD_norm = np.linalg.norm(layer_weights_partialPCD)

            layer_weights_XraySegmPCD = layer_weights[:, half_size:]
            layer_weights_XraySegmPCD_norm = np.linalg.norm(layer_weights_XraySegmPCD)
            if wandb_enabled and wandb_run is not None:
                wandb.log({f"weights_partialPCD/{param}": layer_weights_partialPCD_norm})
                wandb.log({f"weights_XraySegmPCD/{param}": layer_weights_XraySegmPCD_norm})


        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses, use_labelmaps)


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses, use_XraySegm):
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            label, partial_pcd, labelmap, gt = data
            # mean_feature = None
            curr_batch_size = gt.shape[0]


            inputs = partial_pcd[0, :, :].cpu().numpy()
            gt_pcd = gt[0, :, :].cpu().numpy()
            filename = os.path.join('validation_epoch_{:03d}.png'.format(curr_epoch_num))
            if(use_XraySegm):
                labelmap_pcd = labelmap[0,:,:].cpu().numpy()
                plot_pcd_one_view(filename=filename,
                                  pcds=[inputs, labelmap_pcd,gt_pcd],
                                  titles=["input_pcd", "XrayLabelmap","gt"])
            else:
                plot_pcd_one_view(filename=filename,
                                  pcds=[inputs, gt_pcd],
                                  titles=["input_pcd", "gt"])
            if wandb_enabled and wandb_run is not None:
                wandb.log({"inputs_sanitycheck_at_validation_time": wandb.Image(filename)})


            partial_pcd = partial_pcd.float().to(device)
            partial_pcd = partial_pcd.transpose(2, 1).contiguous()

            if(use_XraySegm):
                labelmap = labelmap.float().to(device)
                labelmap = labelmap.transpose(2, 1).contiguous()

            gt = gt.float().to(device)

            result_dict = net(partial_pcd, labelmap, gt, prefix="val")

            # generate pcd images to log to wandb for the first pointcloud in each batch
            #if(curr_epoch_num%10 == 0):
            inputs = result_dict["inputs"][0,:,:].cpu().numpy().T
            fine_pcd = result_dict["result"][0, :, :].cpu().numpy()
            coarse_pcd = result_dict["out1"][0, :, :].cpu().numpy()
            gt_pcd = result_dict["gt"][0, :, :].cpu().numpy()
            filename = os.path.join('validation_epoch_{:03d}.png'.format(curr_epoch_num))
            if(use_XraySegm):
                labelmap_pcd = labelmap[0, :, :].cpu().numpy().T
                plot_pcd_one_view(filename=filename,
                                  pcds=[inputs, labelmap_pcd, coarse_pcd, fine_pcd, gt_pcd],
                                  titles=["input_pcd", "XrayLabelmap", "coarse", "fine", "gt"])
            else:
                plot_pcd_one_view(filename=filename,
                                  pcds=[inputs, coarse_pcd, fine_pcd, gt_pcd],
                                  titles=["input_pcd", "coarse", "fine", "gt"])

            if wandb_enabled and wandb_run is not None:
                wandb.log({"combined_pcds_validation": wandb.Image(filename)})

            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item(), curr_batch_size)

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)

                # save best model locally and to wandb
                model_path = '%s/best_%s_network.pth' % (log_dir, loss_type)
                save_model(model_path, net)

                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)
            if wandb_enabled and wandb_run is not None:
                wandb.log({loss_type: meter.avg})

        logging.info(curr_log)
        logging.info(best_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)

    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    torch.cuda.empty_cache()

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + time


    log_dir = os.path.join(args.work_dir, exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])

    wandb_enabled, wandb_run = setup_wandb(args)

    train()

    if wandb_enabled and wandb_run is not None:
        wandb.finish()


