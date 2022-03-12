"""
Author: Isabella Liu 11/6/21
Feature: Train PSMNet IR reprojection
"""
import gc
import os
import argparse
import numpy as np
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from datasets.messytable_on_real import MessytableOnRealDataset as MessytableDataset
from nets.psmnet import PSMNet
from nets.psmnet_submodule import DisparityRegression
from nets.transformer import Transformer
from nets.discriminator import *
from utils.cascade_metrics import compute_err_metric
from utils.warp_ops import apply_disparity_cu
from utils.reprojection import get_reproj_error_patch
from utils.config import cfg
from utils.reduce import set_random_seed, synchronize, AverageMeterDict, \
    tensor2float, tensor2numpy, reduce_scalar_outputs, make_nograd_func
from utils.util import setup_logger, weights_init, set_requires_grad, \
    adjust_learning_rate, save_scalars, save_scalars_graph, save_images, save_images_grid, disp_error_img

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Reprojection with Pyramid Stereo Network (PSMNet)')
parser.add_argument('--config-file', type=str, default='./configs/local_train_steps.yaml',
                    metavar='FILE', help='Config files')
parser.add_argument('--summary-freq', type=int, default=5, help='Frequency of saving temporary results')
parser.add_argument('--save-freq', type=int, default=1000, help='Frequency of saving checkpoint')
parser.add_argument('--logdir', required=True, help='Directory to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument("--local_rank", type=int, default=0, help='Rank of device in distributed training')
parser.add_argument('--debug', action='store_true', help='Whether run in debug mode (will load less data)')
parser.add_argument('--sub', type=int, default=100, help='If debug mode is enabled, sub will be the number of data loaded')
parser.add_argument('--warp-op', action='store_true',default=True, help='whether use warp_op function to get disparity')
parser.add_argument('--loss-ratio-sim', type=float, default=1, help='Ratio between loss_psmnet_sim and loss_reprojection_sim')
parser.add_argument('--loss_ratio_adversarial', type=float, default=1, help='Ratio for loss_reprojection_real')
parser.add_argument('--gaussian-blur', action='store_true',default=False, help='whether apply gaussian blur')
parser.add_argument('--color-jitter', action='store_true',default=False, help='whether apply color jitter')
parser.add_argument('--ps', type=int, default=11, help='Patch size of doing patch loss calculation')
parser.add_argument('--model', type=str, default='discriminator1')
parser.add_argument('--b1', type=float, default=0)
parser.add_argument('--b2', type=float, default=0.9)
parser.add_argument('--discriminatorlr', type=float, default=0.0001)
parser.add_argument('--gradientpenalty', action='store_true', default=False)
parser.add_argument('--clipc', type=float, default = 0.01)
parser.add_argument('--lam', type=float, default = 10)
parser.add_argument('--diffcv', action='store_true', default=False)

args = parser.parse_args()
cfg.merge_from_file(args.config_file)

# Set random seed to make sure networks in different processes are same
set_random_seed(args.seed)

# Set up distributed training
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1
args.is_distributed = is_distributed
if is_distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group( backend="nccl", init_method="env://")
    synchronize()
cuda_device = torch.device("cuda:{}".format(args.local_rank))

# Set up tensorboard and logger
os.makedirs(args.logdir, exist_ok=True)
os.makedirs(os.path.join(args.logdir, 'models'), exist_ok=True)
summary_writer = tensorboardX.SummaryWriter(logdir=args.logdir)
logger = setup_logger("Reprojection PSMNet", distributed_rank=args.local_rank, save_dir=args.logdir)
logger.info(f'Input args:\n{args}')
logger.info(f'Loaded config file: \'{args.config_file}\'')
logger.info(f'Running with configs:\n{cfg}')
logger.info(f'Running with {num_gpus} GPUs')

# python -m torch.distributed.launch train_psmnet_temporal_ir_reproj.py --summary-freq 1 --save-freq 1 --logdir ../train_10_14_psmnet_ir_reprojection/debug --debug
# python -m torch.distributed.launch train_psmnet_temporal_ir_reproj.py --config-file configs/remote_train_primitive_randscenes.yaml --summary-freq 10 --save-freq 100 --logdir ../train_10_21_psmnet_smooth_ir_reproj/debug --debug


def train(psmnet_model, psmnet_optimizer, discriminator, discriminator_optimizer,
          TrainImgLoader, ValImgLoader, args):
    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # One epoch training loop
        avg_train_scalars_psmnet = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = (len(TrainImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE * num_gpus
            if global_step < 5000:
                dis = True
                adv = True
            elif global_step >= 5000 and global_step < 10000:
                dis = True
                adv = True
            else:
                dis = True
                adv = True

            if global_step > cfg.SOLVER.STEPS:
                break

            # Adjust learning rate
            #adjust_learning_rate(transformer_optimizer, global_step, cfg.SOLVER.LR_CASCADE, cfg.SOLVER.LR_STEPS)
            adjust_learning_rate(psmnet_optimizer, global_step, cfg.SOLVER.LR_CASCADE, cfg.SOLVER.LR_STEPS)
            #adjust_learning_rate(discriminator_optimizer, global_step, cfg.SOLVER.DISCRIMINATOR, cfg.SOLVER.LR_STEPS)


            do_summary = global_step % args.summary_freq == 0
            # Train one sample
            #before_p = discriminator.parameters()
            #before_m = psmnet_model.parameters()
            scalar_outputs_psmnet, img_outputs_psmnet = \
                train_sample(sample, psmnet_model, discriminator, 
                             psmnet_optimizer, discriminator_optimizer, args, isTrain=True, isDis=dis, isAdv=adv)
            #after_p = discriminator.parameters()
            #after_m = psmnet_model.parameters()
            #isSameD = True
            #for p1, p2 in zip(before_p, after_p):
            #    if p1.data.ne(p2.data).sum() > 0:
            #        isSameD = False

            #isSameM = True
            #for p1, p2 in zip(before_m, after_m):
            #    if p1.data.ne(p2.data).sum() > 0:
            #        isSameM = False

            #print(isSameD, isSameM)
            # Save result to tensorboard
            if (not is_distributed) or (dist.get_rank() == 0):
                scalar_outputs_psmnet = tensor2float(scalar_outputs_psmnet)
                avg_train_scalars_psmnet.update(scalar_outputs_psmnet)
                if do_summary:
                    # Update reprojection images
                    #save_images_grid(summary_writer, 'train_reproj', img_output_reproj, global_step, nrow=4)
                    # Update PSMNet images
                    save_images(summary_writer, 'train_psmnet', img_outputs_psmnet, global_step)
                    # Update PSMNet losses
                    scalar_outputs_psmnet.update({'psm_lr': psmnet_optimizer.param_groups[0]['lr']})
                    scalar_outputs_psmnet.update({'dis_lr': discriminator_optimizer.param_groups[0]['lr']})
                    save_scalars(summary_writer, 'train_psmnet', scalar_outputs_psmnet, global_step)

                    total_err_metric_psmnet = avg_train_scalars_psmnet.mean()
                    logger.info(f'Step {global_step} train psmnet: {total_err_metric_psmnet}')

                # Save checkpoints
                if (global_step) % args.save_freq == 0:
                    checkpoint_data = {
                        'epoch': epoch_idx,
                        #'Transformer': transformer_model.state_dict(),
                        'PSMNet': psmnet_model.state_dict(),
                        #'optimizerTransformer': transformer_optimizer.state_dict(),
                        'optimizerPSMNet': psmnet_optimizer.state_dict()
                    }
                    save_filename = os.path.join(args.logdir, 'models', f'model_{global_step}.pth')
                    torch.save(checkpoint_data, save_filename)

                    # Get average results among all batches
                    total_err_metric_psmnet = avg_train_scalars_psmnet.mean()
                    logger.info(f'Step {global_step} train psmnet: {total_err_metric_psmnet}')
        gc.collect()


def train_sample(sample, psmnet_model, discriminator, 
                 psmnet_optimizer, discriminator_optimizer, args, isTrain=True, isDis=True, isAdv=True):
    if isTrain:
        psmnet_model.train()
        discriminator.train()
    else:
        psmnet_model.eval()
        discriminator.eval()

    # Load data
    img_L = sample['img_L'].to(cuda_device)  # [bs, 3, H, W]
    img_R = sample['img_R'].to(cuda_device)
    #img_L_ir_pattern = sample['img_L_ir_pattern'].to(cuda_device)  # [bs, 1, H, W]
    #img_R_ir_pattern = sample['img_R_ir_pattern'].to(cuda_device)

    # Train on simple Transformer
    #img_L_transformed, img_R_transformed = transformer_model(img_L, img_R)  # [bs, 3, H, W]

    # Train on PSMNet
    disp_gt_l = sample['img_disp_l'].to(cuda_device)
    depth_gt = sample['img_depth_l'].to(cuda_device)  # [bs, 1, H, W]
    img_focal_length = sample['focal_length'].to(cuda_device)
    img_baseline = sample['baseline'].to(cuda_device)

    # Resize the 2x resolution disp and depth back to H * W
    # Note this should go before apply_disparity_cu
    disp_gt_l = F.interpolate(disp_gt_l, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]
    depth_gt = F.interpolate(depth_gt, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]

    if args.warp_op:
        img_disp_r = sample['img_disp_r'].to(cuda_device)
        img_disp_r = F.interpolate(img_disp_r, scale_factor=0.5, mode='nearest',
                                   recompute_scale_factor=False)
        disp_gt_l = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
        del img_disp_r

    # Get stereo loss on sim
    mask = (disp_gt_l < cfg.ARGS.MAX_DISP) * (disp_gt_l > 0)  # Note in training we do not exclude bg
    if isTrain:
        pred_disp1, pred_disp2, pred_disp3, cost_vol_1, cost_vol_2, cost_vol = \
                psmnet_model(img_L, img_R)
        
        #dis_output = discriminator(cost_vol)
        sim_pred_disp = pred_disp3
        gt_disp = torch.clone(disp_gt_l).long()
        msk_gt_disp = (gt_disp < cfg.ARGS.MAX_DISP) * (gt_disp > 0)
        #gt_disp[msk_gt_disp] = 0
        if args.diffcv:
            disp_low = torch.floor(disp_gt_l)
            cv_low = F.one_hot(disp_low.long(), num_classes=cfg.ARGS.MAX_DISP).float().permute(0,1,4,2,3)
            disp_up = torch.ceil(disp_gt_l)
            cv_up = F.one_hot(disp_up.long(), num_classes=cfg.ARGS.MAX_DISP).float().permute(0,1,4,2,3)
            x = -(disp_gt_l - disp_up)
            
            b,c,d,h,w = cv_low.shape
            cv_low = cv_low.permute(1,2,0,3,4)
            cv_up = cv_up.permute(1,2,0,3,4)
            x = x.squeeze(1)
            low = cv_low*x
            
            up = cv_up*(1-x)
            low = low.permute(2,0,1,3,4)
            up = up.permute(2,0,1,3,4)
            
            gt_cv = low+up
            gt_cv = gt_cv.cuda()

            ###TEST###
            #dsr = DisparityRegression(cfg.ARGS.MAX_DISP)
            #rec_disp = dsr(gt_cv.squeeze(1))
            #print(rec_disp.shape, disp_gt_l.shape, torch.sum(rec_disp != disp_gt_l))
        else:
            gt_cv = F.one_hot(gt_disp, num_classes=cfg.ARGS.MAX_DISP).float().cuda().permute(0,1,4,2,3)
        cost_vol = cost_vol.unsqueeze(1)
        #print(cost_vol.shape, gt_cv.shape)

        set_requires_grad([discriminator], False)
        #fake_out = discriminator(cost_vol)
        #real_out = discriminator(gt_cv)
        #print(torch.sum(fake_out <= 1))
        #assert torch.sum(fake_out <= 1) == 2*12*16*32
        #loss_discriminator = - torch.sum((1-fake_out).log()) - torch.sum(real_out.log())
        #print('require: ',loss_discriminator.requires_grad)
        #print(loss_discriminator.item())
        #print(fake_out.shape, real_out.shape)
        #with torch.no_grad():
        pred_out = discriminator(cost_vol)
        #print(pred_out.shape)
        if args.model == 'discriminator1' or args.model == 'discriminator2':
            if args.gradientpenalty:
                loss_adversarial = torch.mean(-pred_out)
            else:
                loss_adversarial = F.binary_cross_entropy_with_logits(pred_out, torch.ones(1).expand_as(pred_out).cuda())
        elif args.model == 'critic1' or args.model == 'critic2':
            loss_adversarial = torch.mean(pred_out)

        #print(gt_disp.shape, gt_cv.shape, cost_vol.shape)
        #loss_psmnet = F.binary_cross_entropy(cost_vol, gt_cv, reduction = 'mean')
        loss_psmnet = 0.5 * F.smooth_l1_loss(pred_disp1[mask], disp_gt_l[mask], reduction='mean') \
               + 0.7 * F.smooth_l1_loss(pred_disp2[mask], disp_gt_l[mask], reduction='mean') \
               + F.smooth_l1_loss(pred_disp3[mask], disp_gt_l[mask], reduction='mean')

        if isAdv:
            sim_loss = loss_psmnet * args.loss_ratio_sim + loss_adversarial * args.loss_ratio_adversarial
        else:
            sim_loss = loss_psmnet * args.loss_ratio_sim

        #transformer_optimizer.zero_grad()
        psmnet_optimizer.zero_grad()
        sim_loss.backward()
        psmnet_optimizer.step()
        #transformer_optimizer.step()


        set_requires_grad([discriminator], True)
        if isDis:
            discriminator_optimizer.zero_grad()
        cost_vol = cost_vol.detach()

        print(cost_vol.shape, gt_cv.shape)
        

        if args.model == 'discriminator1' or args.model == 'discriminator2':
            if args.gradientpenalty:
                
                cost_vol_clone = torch.clone(cost_vol)
                cost_vol_clone.requires_grad = True

                epsilon = torch.rand(1).to(cuda_device)
                cost_vol_hat = epsilon*gt_cv + (1-epsilon)*cost_vol_clone

                fake_out_d = discriminator(cost_vol_clone)#.unsqueeze(5)
                loss_fake_d = torch.mean(fake_out_d).unsqueeze(0)
                #print(loss_fake_d.unsqueeze(0).shape, fake_out_d.shape, one.shape)
                loss_fake_d.backward()

                real_out_d = discriminator(gt_cv)#.unsqueeze(5)
                loss_real_d = -torch.mean(real_out_d).unsqueeze(0)
                loss_real_d.backward()

                cost_vol_hat = torch.autograd.Variable(cost_vol_hat, requires_grad=True)
                disc_out = discriminator(cost_vol_hat)

                gradient = torch.autograd.grad(disc_out, cost_vol_hat, grad_outputs=torch.ones_like(disc_out).to(cuda_device), create_graph=True, 
                        retain_graph=True, only_inputs=True)
                gradient = torch.reshape(gradient[0],(2,1,192,-1))
                gnorm = torch.linalg.matrix_norm(gradient)
                
                #print(fake_out_d.shape, real_out_d.shape, gnorm.shape, gradient[0].shape)
                loss_gradient = args.lam*torch.mean(torch.pow((gnorm - 1),2))
                loss_gradient.backward()

                loss_discriminator = loss_fake_d + loss_real_d + loss_gradient
           
            else:
                fake_out_d = discriminator(cost_vol)#.unsqueeze(5)

                real_out_d = discriminator(gt_cv)#.unsqueeze(5)

                loss_d_fake = F.binary_cross_entropy(fake_out_d, torch.zeros(1).expand_as(fake_out_d).cuda())
                loss_d_real = F.binary_cross_entropy(real_out_d, torch.ones(1).expand_as(real_out_d).cuda())
                print(torch.sum(real_out_d).item(), torch.sum(fake_out_d).item(), real_out_d.shape)
                #print(torch.sum(cost_vol), torch.sum(gt_cv), cost_vol.shape, gt_cv.shape)

                loss_discriminator = (loss_d_fake + loss_d_real) * 0.5
                loss_discriminator.backward()
        
        elif args.model == 'critic1' or args.model == 'critic2':
            fake_out_d = discriminator(cost_vol)#.unsqueeze(5)
            loss_fake_d = -torch.mean(fake_out_d).unsqueeze(0)
            loss_fake_d.backward()

            real_out_d = discriminator(gt_cv)#.unsqueeze(5)
            loss_real_d = torch.mean(real_out_d).unsqueeze(0)
            loss_real_d.backward()

            loss_discriminator = loss_fake_d - loss_real_d
            w_d = loss_real_d - loss_fake_d
        
        else:
            raise Exception(NotImplementedError)

        if isDis:
            #for param in discriminator.parameters():
            #    print(param.grad.data.sum())
            discriminator_optimizer.step()
            if args.model == 'critic1' or args.model == 'critic2':
                with torch.no_grad():
                    for param in discriminator.parameters():
                        param.clamp_(-args.clipc, args.clipc)

    else:
        with torch.no_grad():
            pred_disp = psmnet_model(img_L, img_R, img_L_transformed, img_R_transformed)
            loss_psmnet = F.smooth_l1_loss(pred_disp[mask], disp_gt_l[mask], reduction='mean')
            sim_loss = loss_psmnet

    # Get reprojection loss on sim_ir_pattern
    #sim_ir_reproj_loss, sim_ir_warped, sim_ir_reproj_mask = get_reproj_error_patch(
    #    input_L=img_L_ir_pattern,
    #    input_R=img_R_ir_pattern,
    #    pred_disp_l=sim_pred_disp,
    #    mask=mask,
    #    ps=args.ps
    #)

    # Backward on sim_ir_pattern reprojection


    """
    # Get reprojection loss on real
    img_real_L = sample['img_real_L'].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_R = sample['img_real_R'].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_L_ir_pattern = sample['img_real_L_ir_pattern'].to(cuda_device)
    img_real_R_ir_pattern = sample['img_real_R_ir_pattern'].to(cuda_device)
    img_real_L_transformed, img_real_R_transformed = transformer_model(img_real_L, img_real_R)  # [bs, 3, H, W]
    if isTrain:
        pred_disp1, pred_disp2, pred_disp3 = psmnet_model(img_real_L, img_real_R, img_real_L_transformed, img_real_R_transformed)
        real_pred_disp = pred_disp3
    else:
        with torch.no_grad():
            real_pred_disp = psmnet_model(img_real_L, img_real_R, img_real_L_transformed, img_real_R_transformed)
    real_ir_reproj_loss, real_ir_warped, real_ir_reproj_mask = get_reproj_error_patch(
        input_L=img_real_L_ir_pattern,
        input_R=img_real_R_ir_pattern,
        pred_disp_l=real_pred_disp,
        ps=args.ps
    )

    # Backward on real
    real_loss = real_ir_reproj_loss * args.loss_ratio_real
    if isTrain:
        transformer_optimizer.zero_grad()
        psmnet_optimizer.zero_grad()
        real_loss.backward()
        psmnet_optimizer.step()
        transformer_optimizer.step()
    """
    # Save reprojection outputs and images
    #img_output_reproj = {
    #    'sim_reprojection': {
    #        'target': img_L_ir_pattern, 'warped': sim_ir_warped, 'pred_disp': sim_pred_disp, 'mask': sim_ir_reproj_mask
    #    }
    #    #'real_reprojection': {
    #    #    'target': img_real_L_ir_pattern, 'warped': real_ir_warped, 'pred_disp': real_pred_disp, 'mask': real_ir_reproj_mask
    #    #}
    #}

    # Compute stereo error metrics on sim
    pred_disp = sim_pred_disp
    scalar_outputs_psmnet = {'loss': loss_psmnet.item(),
                             'dis_loss': loss_discriminator.item(),
                             'adv_loss': loss_adversarial.item()
                             #'sim_reprojection_loss': sim_ir_reproj_loss.item()
                             #'real_reprojection_loss': real_ir_reproj_loss.item()
                            }
    if args.model == 'critic1' or args.model == 'critic2':
        scalar_outputs_psmnet['w_distance'] = w_d.item()
    err_metrics = compute_err_metric(disp_gt_l,
                                     depth_gt,
                                     pred_disp,
                                     img_focal_length,
                                     img_baseline,
                                     mask)
    scalar_outputs_psmnet.update(err_metrics)
    # Compute error images
    pred_disp_err_np = disp_error_img(pred_disp[[0]], disp_gt_l[[0]], mask[[0]])
    pred_disp_err_tensor = torch.from_numpy(np.ascontiguousarray(pred_disp_err_np[None].transpose([0, 3, 1, 2])))
    img_outputs_psmnet = {
        'disp_gt_l': disp_gt_l[[0]].repeat([1, 3, 1, 1]),
        'disp_pred': pred_disp[[0]].repeat([1, 3, 1, 1]),
        'disp_err': pred_disp_err_tensor,
        'input_L': img_L,
        'input_R': img_R
    }

    if is_distributed:
        scalar_outputs_psmnet = reduce_scalar_outputs(scalar_outputs_psmnet, cuda_device)
    return scalar_outputs_psmnet, img_outputs_psmnet#, img_output_reproj


if __name__ == '__main__':
    # Obtain dataloader
    train_dataset = MessytableDataset(cfg.SPLIT.TRAIN, gaussian_blur=args.gaussian_blur, color_jitter=args.color_jitter,
                                    debug=args.debug, sub=args.sub, onreal = False)
    val_dataset = MessytableDataset(cfg.SPLIT.VAL, gaussian_blur=args.gaussian_blur, color_jitter=args.color_jitter,
                                    debug=args.debug, sub=10, onreal = False)
    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=dist.get_world_size(),
                                                          rank=dist.get_rank())

        TrainImgLoader = torch.utils.data.DataLoader(train_dataset, cfg.SOLVER.BATCH_SIZE, sampler=train_sampler,
                                                     num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True, pin_memory=True)
        ValImgLoader = torch.utils.data.DataLoader(val_dataset, cfg.SOLVER.BATCH_SIZE, sampler=val_sampler,
                                                   num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False, pin_memory=True)
    else:
        TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                     shuffle=True, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True)

        ValImgLoader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                   shuffle=False, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False)

    # Create Transformer model


    # Create PSMNet model
    psmnet_model = PSMNet(maxdisp=cfg.ARGS.MAX_DISP, loss='BCE', transform=False).to(cuda_device)
    psmnet_optimizer = torch.optim.Adam(psmnet_model.parameters(), lr=cfg.SOLVER.LR_CASCADE, betas=(0.9, 0.999))
    if is_distributed:
        psmnet_model = torch.nn.parallel.DistributedDataParallel(
            psmnet_model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        psmnet_model = torch.nn.DataParallel(psmnet_model)

    if args.model == 'discriminator1':
        discriminator = Discriminator(inplane=1, outplane=1).to(cuda_device)
    elif args.model == 'discriminator2':
        discriminator = Discriminator2(inplane=1, outplane=1).to(cuda_device)
    elif args.model == 'critic1':
        discriminator = Critic(inplane=1, outplane=1).to(cuda_device)
    elif args.model == 'critic2':
        discriminator = Critic2(inplane=1, outplane=1).to(cuda_device)
    else:
        raise Exception(NotImplementedError)
    #discriminator = NLayerDiscriminator()
    if args.model == 'critic1' or args.model == 'critic2':
        discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=args.discriminatorlr)
    elif args.model == 'discriminator1' or args.model == 'discriminator2':
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminatorlr, betas=(args.b1, args.b2))
    else:
        raise Exception(NotImplementedError)
    if is_distributed:
        discriminator = torch.nn.parallel.DistributedDataParallel(
            discriminator, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        discriminator = torch.nn.DataParallel(discriminator)

    #psm_param = sum(p.numel() for p in psmnet_model.parameters() if p.requires_grad)
    #trans_param = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
    #print(str(psm_param)+','+str(trans_param)+','+str(psm_param + trans_param))
    

    # Start training
    train(psmnet_model, psmnet_optimizer, discriminator, discriminator_optimizer, TrainImgLoader, ValImgLoader, args)