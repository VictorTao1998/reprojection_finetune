import torch
import numpy as np
import torch.nn.functional as F

a = np.linspace(0,4,5)
w = 256
h = 512
d = 192
x = cost_p = np.array(list(map(list, list(np.ndindex(d,w,h))))).reshape(d,w,h,3)
print(x)
#result = np.zeros([d,w,h,3])
#result[:,:,:,2] = z


    
    


#print(result)


