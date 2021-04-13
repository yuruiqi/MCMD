import numpy as np
from monai.transforms import *
import json
import matplotlib.pyplot as plt

from ImageProcess.Operations import get_box

with open('/homes/rqyu/Projects/MCMD/Data/Split Temp/train.json') as f:
    trainlist = json.load(f)
with open('/homes/rqyu/Projects/MCMD/Data/Split Temp/val.json') as f:
    vallist = json.load(f)
with open('/homes/rqyu/Projects/MCMD/Data/Split Temp/test.json') as f:
    testlist = json.load(f)

datalist = trainlist + vallist + testlist

hs = []
ds = []
ws = []
for dict in datalist:
    data = LoadNiftid(keys=['image', 'mask'])(dict)

    data = AddChanneld(keys=['image', 'mask'])(data)
    data = Spacingd(keys=['image', 'mask'], pixdim=[0.7, 0.7, 1.25], mode=['bilinear', 'nearest'])(data)

    box = get_box(data['mask'][0], norm=False)
    hs.append(box[3] - box[0])
    ws.append(box[4] - box[1])
    ds.append(box[5] - box[2])

plt.hist(hs)
plt.show()
plt.hist(ws)
plt.show()
plt.hist(ds)
plt.show()
