def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic=np.eye(4)):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    return world_points


def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(np.float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid

def disp2pcd(pred_disp, img_focal_length, img_baseline, cam_intrinsic, cam_extrinsic)
    
    pred_depth = img_focal_length * img_baseline / pred_disp
    pred_depth_np = pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()


    pred_pts = depth2pts_np(pred_depth_np, cam_intrinsic, cam_extrinsic)
    pred_pts = np.reshape(pred_pts, [pred_pts.shape[0], pred_pts.shape[1]])

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_pts)
    return pred_pcd