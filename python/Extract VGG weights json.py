#!/usr/bin/env python

import torch
import torchvision.models as models


vgg16 = models.vgg16(pretrained=True)


# In[2]:


import numpy as np
import os
filename = "../weights/vgg16-weights.bin"
try:
    os.remove(filename)
except:
    print("file did not exisist")
f = open(filename, 'wb')
for k, v in vgg16.state_dict().items():
    print("saving "+k)
    v.require_grad = False
    binary_format = v.numpy().tobytes('C')
    f.write(binary_format)
f.close()


