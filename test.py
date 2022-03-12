import torch
import numpy as np
import torch.nn.functional as F

cv = np.zeros([4,2,2])
disp = torch.tensor([[1.3,2.6],[3,0.2]])


disp_low = torch.floor(disp)
cv_low = F.one_hot(disp_low.long(), num_classes=4).float().permute(2,0,1)


disp_up = torch.ceil(disp)
cv_up = F.one_hot(disp_up.long(), num_classes=4).float().permute(2,0,1)



x = -(disp - disp_up)
low = cv_low*x
up = cv_up*(1-x)
gt_cv = low+up
print(gt_cv.permute(1,2,0))

maxdisp = 4

