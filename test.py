import torch
import numpy as np
import torch.nn.functional as F

maxdepth = 0.2
offset = 0.01
depth_value = torch.arange(offset, offset+maxdepth, offset).float()
n_depth = depth_value.shape[0]
#print(depth_value)
depth = torch.tensor([[0.013,0.035],[0.064,0.185]]).unsqueeze(0)
#print(depth.shape)


#mask_pe = np.logical_and(depth_vd < offset, depth_vd >=0).type(torch.bool)
#mask_ne = np.logical_and(depth_vd > -offset, depth_vd <=0).type(torch.bool)
#mask_e = np.logical_and(mask_pe, mask_ne).type(torch.bool)
#print(mask_pe.shape)
depth_low_ind = (torch.div(depth, offset, rounding_mode='floor') - 1).type(torch.long)
depth_high_ind = depth_low_ind + 1
mask_pe = F.one_hot(depth_low_ind,num_classes=n_depth).permute(3,0,1,2).squeeze(1).type(torch.bool)
mask_ne = F.one_hot(depth_high_ind,num_classes=n_depth).permute(3,0,1,2).squeeze(1).type(torch.bool)
#print(depth_low_ind.shape, mask_pe.shape)
#assert 1==0
depth_low = depth - torch.remainder(depth, offset)

x_low = (offset + depth_low -depth)/offset
x_high = 1 - x_low

depth_out = torch.zeros([20,2,2])
#print(depth_out.shape, depth_out[mask_pe].shape)
depth_out[mask_pe] = x_low.flatten()
depth_out[mask_ne] = x_high.flatten()
#depth_out[mask_e] = 1.
#mask_e = np.logical_and(mask_pe, mask_ne)
#mask_p = np.logical_xor(mask_pe, mask_e)
#mask_n = np.logical_xor(mask_ne, mask_e)
depth_value = depth_value[:, None, None]
#print(depth_out)
out = torch.sum(depth_out * depth_value, 0, keepdim=True)
print(out)

#n_mask = np.logical_not(mask)

#depth_v[mask] = 1
#depth_v[n_mask] = 0
#print(depth_value)
#print(depth_v, depth_v.shape)
#result = np.zeros([d,w,h,3])
#result[:,:,:,2] = z

cv = torch.rand(4,2,2)
disp = torch.arange(0, 4).float()[:,None,None]
disp1 = torch.arange(0, 4, 0.5).float()[:,None,None]
#print(disp1.shape, cv.shape)
sm = torch.nn.Softmax(dim=0)
sm_cv = sm(cv)

cvh = cv[None, None,:,:,:]
cv1 = F.interpolate(cvh, (8, 2, 2), mode='trilinear', align_corners=False)[0,0,:,:,:]
#print(cv1.shape)
sm_cv1 = sm(cv1)
#print(sm_cv.shape)
out = torch.sum(sm_cv * disp, 0, keepdim=True)
out1 = torch.sum(sm_cv1 * disp1, 0, keepdim=True)
#print(out,out1)



