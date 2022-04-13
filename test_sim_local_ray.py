"""
Author: Isabella Liu 10/14/21
Feature: Test PSMNet Reprojection
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import cm
from datasets.messytable_test_local import get_test_loader
import open3d as o3d
from utils.np_utils import *
# from nets.psmnet import PSMNet
from nets.psmnet_ray import PSMNet, Renderer
from nets.transformer import Transformer
from utils.cascade_metrics import compute_err_metric, compute_obj_err
from utils.config import cfg
from utils.test_util import (
    load_from_dataparallel_model,
    save_gan_img,
    save_img,
    save_obj_err_file,
    save_prob_volume,
)
from utils.util import depth_error_img, disp_error_img, get_time_string, setup_logger
from utils.warp_ops import apply_disparity_cu
from nets.psmnet_submodule_ray import *

parser = argparse.ArgumentParser(description="Testing for Reprojection + PSMNet")
parser.add_argument(
    "--config-file",
    type=str,
    default="./configs/local_test.yaml",
    metavar="FILE",
    help="Config files",
)
parser.add_argument(
    "--model", type=str, default="", metavar="FILE", help="Path to test model"
)
parser.add_argument(
    "--gan-model", type=str, default="", metavar="FILE", help="Path to test gan model"
)
parser.add_argument(
    "--output",
    type=str,
    help="Path to output folder",
    required=True,
)
parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")
parser.add_argument(
    "--annotate", type=str, default="", help="Annotation to the experiment"
)
parser.add_argument(
    "--onreal", action="store_true", default=False, help="Test on real dataset"
)
parser.add_argument(
    "--analyze-objects",
    action="store_true",
    default=True,
    help="Analyze on different objects",
)
parser.add_argument(
    "--exclude-bg",
    action="store_true",
    default=False,
    help="Exclude background when testing",
)
parser.add_argument(
    "--warp-op",
    action="store_true",
    default=True,
    help="Use warp_op function to get disparity",
)
parser.add_argument(
    "--exclude-zeros",
    action="store_true",
    default=False,
    help="Whether exclude zero pixels in realsense",
)
parser.add_argument(
    "--local_rank", type=int, default=0, help="Rank of device in distributed training"
)
parser.add_argument(
    "--cv", action="store_true", default=False
)



args = parser.parse_args()
cfg.merge_from_file(args.config_file)
cuda_device = torch.device("cuda:{}".format(args.local_rank))
# If path to gan model is not specified, use gan model from cascade model

if args.gan_model == "":
    args.gan_model = args.model

# Calculate error for real and 3D printed objects
real_obj_id = [4, 5, 7, 9, 13, 14, 15, 16]



# python test_psmnet_with_confidence.py --model /code/models/model_4.pth --onreal --exclude-bg --exclude-zeros
# python test_psmnet_with_confidence.py --config-file configs/remote_test.yaml --model ../train_8_14_cascade/train1/models/model_best.pth --onreal --exclude-bg --exclude-zeros --debug --gan-model


def test(psmnet_model, render_model, rayreg_model, val_loader, logger, log_dir):
    #transformer_model.eval()
    psmnet_model.eval()
    render_model.eval()

    total_err_metrics = {
        "epe": 0,
        "bad1": 0,
        "bad2": 0,
        "depth_abs_err": 0,
        "depth_err2": 0,
        "depth_err4": 0,
        "depth_err8": 0,
        "normal_err": 0,
    }

    total_obj_disp_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_4_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_normal_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_count = np.zeros(cfg.SPLIT.OBJ_NUM)
    os.mkdir(os.path.join(log_dir, "pred_disp"))
    os.mkdir(os.path.join(log_dir, "cost_vol_pcd"))
    os.mkdir(os.path.join(log_dir, "gt_disp"))
    os.mkdir(os.path.join(log_dir, "pred_disp_abs_err_cmap"))
    os.mkdir(os.path.join(log_dir, "pred_depth"))
    os.mkdir(os.path.join(log_dir, "gt_depth"))
    os.mkdir(os.path.join(log_dir, "pred_depth_abs_err_cmap"))
    os.mkdir(os.path.join(log_dir, "pred_conf"))
    os.mkdir(os.path.join(log_dir, "pred_pcd"))
    os.mkdir(os.path.join(log_dir, "gt_pcd"))
    os.mkdir(os.path.join(log_dir, "realsense_pcd"))
    os.mkdir(os.path.join(log_dir, "prob_volume"))

    for iteration, data in enumerate(tqdm(val_loader)):
        
        img_L = data["img_L"].cuda()  # [bs, 1, H, W]
        img_R = data["img_R"].cuda()

        img_disp_l = data["img_disp_l"].cuda()
        img_depth_l = data["img_depth_l"].cuda()
        img_depth_realsense = data["img_depth_realsense"].cuda()
        img_label = data["img_label"].cuda()
        img_focal_length = data["focal_length"].cuda()
        img_baseline = data["baseline"].cuda()
        prefix = data["prefix"][0]
        #robot_mask = data["robot_mask"].cuda()
        cam_intrinsic = data["intrinsic_l"].numpy()
        cam_intrinsic[:, :2] /= 2

        # Note(rayc): the resize and padding should be done in the dataloader,along with cam_intrinsic
        img_L = F.interpolate(
            img_L, (540, 960), mode="bilinear", recompute_scale_factor=False
        )
        img_R = F.interpolate(
            img_R, (540, 960), mode="bilinear", recompute_scale_factor=False
        )

        img_disp_l = F.interpolate(
            img_disp_l, (540, 960), mode="nearest", recompute_scale_factor=False
        )
        img_depth_l = F.interpolate(
            img_depth_l, (540, 960), mode="nearest", recompute_scale_factor=False
        )
        img_depth_realsense = F.interpolate(
            img_depth_realsense,
            (540, 960),
            mode="nearest",
            recompute_scale_factor=False,
        )
        img_label = F.interpolate(
            img_label, (540, 960), mode="nearest", recompute_scale_factor=False
        ).type(torch.int)
        #img_robot_mask = F.interpolate(
        #    robot_mask, (540, 960), mode="nearest", recompute_scale_factor=False
        #).type(torch.int)

        # If using warp_op, computing img_disp_l from img_disp_r
        if args.warp_op:
            img_disp_r = data["img_disp_r"].cuda()
            img_depth_r = data["img_depth_r"].cuda()
            img_disp_r = F.interpolate(
                img_disp_r, (540, 960), mode="nearest", recompute_scale_factor=False
            )
            img_depth_r = F.interpolate(
                img_depth_r, (540, 960), mode="nearest", recompute_scale_factor=False
            )
            img_disp_l = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))
            img_depth_l = apply_disparity_cu(
                img_depth_r, img_disp_r.type(torch.int)
            )  # [bs, 1, H, W]

        # If test on real dataset need to crop input image to (540, 960)
        if args.onreal:
            img_L = F.interpolate(
                img_L,
                (540, 960),
                mode="bilinear",
                recompute_scale_factor=False,
                align_corners=False,
            )
            img_R = F.interpolate(
                img_R,
                (540, 960),
                mode="bilinear",
                recompute_scale_factor=False,
                align_corners=False,
            )

        #with torch.no_grad():
        #    img_L_transformed, img_R_transformed = transformer_model(img_L, img_R)

        # Pad the imput image and depth disp image to 960 * 544
        right_pad = 0
        top_pad = 4
        img_L = F.pad(
            img_L, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode="constant", value=0
        )
        
        img_R = F.pad(
            img_R, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode="constant", value=0
        )

        """
        img_L_transformed = F.pad(
            img_L_transformed,
            (0, right_pad, top_pad, 0, 0, 0, 0, 0),
            mode="constant",
            value=0,
        )
        img_R_transformed = F.pad(
            img_R_transformed,
            (0, right_pad, top_pad, 0, 0, 0, 0, 0),
            mode="constant",
            value=0,
        )
        """

        #robot_mask = img_robot_mask == 0
        if args.exclude_bg:
            # Mask ground pixel to False
            img_ground_mask = (img_depth_l > 0) & (img_depth_l < 1.25)
            mask = (
                (img_disp_l < cfg.ARGS.MAX_DISP)
                * (img_disp_l > 0)
                * img_ground_mask
                #* robot_mask
            )
        else:
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0) #* robot_mask

        # Exclude uncertain pixel from realsense_depth_pred
        realsense_zeros_mask = img_depth_realsense > 0
        if args.exclude_zeros:
            mask = mask * realsense_zeros_mask
        mask = mask.type(torch.bool)

        ground_mask = (
            torch.logical_not(mask).squeeze(0).squeeze(0).detach().cpu().numpy()
        )

        with torch.no_grad():#, pred_conf, cost
            cost_vol = psmnet_model(
                img_L, img_R
            )
            B,C,D,H,W = cost_vol.shape
            
            ray = cost_vol.view(B,C,D,H*W)  
            #print(ray.shape)
            pred_disp = torch.zeros([1,1,H*W]).to(cuda_device)
            #print(pred_disp.shape)
            n = 10240

            for i in range(int(H*W/n)):
                ray_c = ray[:,:,:,i*n:(i+1)*n]
                disp_candidate = torch.arange(0,cfg.ARGS.MAX_DISP).float().to(cuda_device).expand(n,-1).permute(1,0)+0.5
                disp_candidate_g = ((disp_candidate/192.) * 2.)-1.
                D = 192

                disp_cand_n = torch.arange(0, n).float().to(cuda_device).expand(192,-1)
            
                disp_cand_grid = torch.cat((disp_candidate_g[...,None], disp_cand_n[...,None]), -1).view(B,D,n,2)

                pts_feat = F.grid_sample(ray_c, disp_cand_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                
                disp_fea = disp_candidate_g[None,None,:,:]
                pts_feat = torch.cat((pts_feat, disp_fea), 1).view(B,D*n,C+1)

                alpha = render_model(pts_feat)
                output = alpha.view(B,D,n,1)
                #print(output.shape, disp_candidate.shape)
                pred_disp_c = rayreg_model(output, disp_candidate[None,:,:,None]).view(B,1,n)
                pred_disp[0,0,i*n:(i+1)*n] = pred_disp_c

            pred_disp = pred_disp.view(1,1,H,W)

        pred_disp = pred_disp[
            :, :, top_pad:, :
        ]  # TODO: if right_pad > 0 it needs to be (:-right_pad)
        #pred_conf = pred_conf[:, :, top_pad:, :]
        #pred_conf = pred_conf.detach().cpu().numpy()[0, 0]

        pred_depth = img_focal_length * img_baseline / pred_disp  # pred depth in m


        # Get loss metric
        
        err_metrics = compute_err_metric(
            img_disp_l, img_depth_l, pred_disp, img_focal_length, img_baseline, mask, isdisp=True
        )
        for k in total_err_metrics.keys():
            if k != 'normal_err':
                total_err_metrics[k] += err_metrics[k]
        

        # Get disparity image
        pred_disp_np = pred_disp.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
        # pred_disp_np[ground_mask] = -1  # Note(rayc): better save the raw output

        # Get disparity ground truth image
        gt_disp_np = img_disp_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        # gt_disp_np[ground_mask] = -1

        # Get disparity error image
        pred_disp_err_np = disp_error_img(pred_disp, img_disp_l, mask)

        # Get depth image
        pred_depth_np = (
            pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()
        )  # in m, [H, W]
        # crop depth map to [0.2m, 2m]
        # pred_depth_np[pred_depth_np < 0.2] = -1
        # pred_depth_np[pred_depth_np > 2] = -1
        #pred_depth_np[ground_mask] = -1

        # Get depth ground truth image
        gt_depth_np = img_depth_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        #gt_depth_np[ground_mask] = -1

        # Get depth error image
        pred_depth_err_np = depth_error_img(pred_depth * 1000, img_depth_l * 1000, mask)

        # TODO get realsense images
        realsense_depth_np = img_depth_realsense.squeeze(0).squeeze(0).detach().cpu().numpy()

        # Save images
        angle = save_img(
            log_dir,
            prefix,
            pred_disp_np,
            gt_disp_np,
            pred_disp_err_np,
            pred_depth_np,
            gt_depth_np,
            realsense_depth_np,
            pred_depth_err_np,
            #pred_conf,
            mask,
            cam_intrinsic=cam_intrinsic,
            isdisp=True
        )
        #print(np.sum(np.isnan(angle)))
        
        #print(angle.shape)
        angle = np.reshape(angle, (gt_depth_np.shape[-2],gt_depth_np.shape[-1]))
        #print(angle.shape, mask.shape)
        angle_err = np.mean(angle[mask.cpu()[0,0]])
        total_err_metrics['normal_err'] += angle_err
        err_metrics['normal_err'] = angle_err

        # Get object error
        obj_disp_err, obj_depth_err, obj_depth_4_err, obj_normal_err, obj_count = compute_obj_err(
            img_disp_l,
            img_depth_l,
            pred_disp,
            angle,
            img_focal_length,
            img_baseline,
            img_label,
            mask,
            cfg.SPLIT.OBJ_NUM,
        )
        total_obj_disp_err += obj_disp_err
        total_obj_depth_err += obj_depth_err
        total_obj_depth_4_err += obj_depth_4_err
        total_obj_normal_err += obj_normal_err 
        total_obj_count += obj_count

        
        logger.info(f"Test instance {prefix} - {err_metrics}")



        # save cost volume
        #prob_volume = cost[0].detach().cpu().numpy()
        #save_prob_volume(prob_volume, log_dir, prefix)

    # Get final error metrics
    for k in total_err_metrics.keys():
        total_err_metrics[k] /= len(val_loader)
    logger.info(f"\nTest on {len(val_loader)} instances\n {total_err_metrics}")

    # Save object error to csv file
    total_obj_disp_err /= total_obj_count
    total_obj_depth_err /= total_obj_count
    total_obj_depth_4_err /= total_obj_count
    total_obj_normal_err /= total_obj_count
    save_obj_err_file(
        total_obj_disp_err, total_obj_depth_err, total_obj_depth_4_err, total_obj_normal_err, log_dir
    )

    logger.info(f"Successfully saved object error to obj_err.txt")

    # Get error on real and 3d printed objects
    real_depth_error = 0
    real_depth_error_4mm = 0
    printed_depth_error = 0
    printed_depth_error_4mm = 0
    for i in range(cfg.SPLIT.OBJ_NUM):
        if i in real_obj_id:
            real_depth_error += total_obj_depth_err[i]
            real_depth_error_4mm += total_obj_depth_4_err[i]
        else:
            printed_depth_error += total_obj_depth_err[i]
            printed_depth_error_4mm += total_obj_depth_4_err[i]
    real_depth_error /= len(real_obj_id)
    real_depth_error_4mm /= len(real_obj_id)
    printed_depth_error /= cfg.SPLIT.OBJ_NUM - len(real_obj_id)
    printed_depth_error_4mm /= cfg.SPLIT.OBJ_NUM - len(real_obj_id)

    logger.info(
        f"Real objects - absolute depth error: {real_depth_error}, depth 4mm: {real_depth_error_4mm} \n"
        f"3D printed objects - absolute depth error {printed_depth_error}, depth 4mm: {printed_depth_error_4mm}"
    )


def main():
    # Obtain the dataloader
    val_loader = get_test_loader(cfg.SPLIT.VAL, args.debug, sub=40, onReal=args.onreal)

    # Tensorboard and logger
    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, f"{get_time_string()}_{args.annotate}")
    os.mkdir(log_dir)
    logger = setup_logger(
        "Reprojection-PSMNet Testing", distributed_rank=0, save_dir=log_dir
    )
    logger.info(f"Annotation: {args.annotate}")
    logger.info(f"Input args {args}")
    logger.info(f"Loaded config file '{args.config_file}'")
    logger.info(f"Running with configs:\n{cfg}")

    # Get cascade model
    logger.info(f"Loaded the checkpoint: {args.model}")
    #transformer_model = Transformer().to(cuda_device)
    psmnet_model = PSMNet(maxdisp=cfg.ARGS.MAX_DISP, transform=False).to(cuda_device)
    render_model = Renderer(D=8, W=256, input_ch=1, output_ch=1, input_ch_feat=8, skips=[4]).to(cuda_device)
    rayreg_model = RayRegression().to(cuda_device)

    #transformer_model_dict = load_from_dataparallel_model(args.model, "Transformer")
    #transformer_model.load_state_dict(transformer_model_dict)
    psmnet_model_dict = load_from_dataparallel_model(args.model, "PSMNet")
    psmnet_model.load_state_dict(psmnet_model_dict)
    render_model_dict = load_from_dataparallel_model(args.model, "Render")
    render_model.load_state_dict(render_model_dict)
    test(psmnet_model, render_model, rayreg_model, val_loader, logger, log_dir)


if __name__ == "__main__":
    main()