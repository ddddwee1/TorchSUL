import torch
from config import cfg
from config import update_config
from TorchSUL import Model as M
import models.hrnet_dekr
import loss 
import hrnet

update_config(cfg)
model_dnet = models.hrnet_dekr.get_pose_net(cfg, is_train=False)
model = loss.ModelWithLoss(model_dnet)
M.Saver(model).restore('./model/', strict=False)
# print(model.model)
model = model.model
model.eval()

net_dekr = hrnet.DEKR(17)

x = torch.ones(1, 3, 512, 512)
net_dekr(x)
net_dekr.eval()
net_dekr.bn_eps(1e-5)
net = net_dekr.backbone

source_params = {'other':[], 'fuse':[]}
target_params = {'other':[], 'fuse':[]}
source_buffs = {'other':[], 'fuse':[]}
target_buffs = {'other':[], 'fuse':[]}

for p in net.named_parameters():
    name = p[0]
    if 'fuse' in name:
        target_params['fuse'].append(p)
    else:
        target_params['other'].append(p)

for p in net.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    if 'fuse' in name:
        target_buffs['fuse'].append(p)
    else:
        target_buffs['other'].append(p)

for p in model.named_parameters():
    name = p[0]
    if 'heatmap' in name:
        continue 
    if 'offset' in name:
        continue 
    if 'fuse' in name:
        source_params['fuse'].append(p)
    else:
        source_params['other'].append(p)

for p in model.named_buffers():
    name = p[0]
    if 'heatmap' in name:
        continue 
    if 'offset' in name:
        continue 
    if 'tracked' in name:
        continue
    if 'fuse' in name:
        source_buffs['fuse'].append(p)
    else:
        source_buffs['other'].append(p)

for ps, pt in zip(source_params['fuse'], target_params['fuse']):
    # print(pt[0], ps[0])
    pt[1].data[:] = ps[1].data[:]
for ps, pt in zip(source_params['other'], target_params['other']):
    # print(pt[0], ps[0])
    pt[1].data[:] = ps[1].data[:]
for ps, pt in zip(source_buffs['fuse'], target_buffs['fuse']):
    # print(pt[0], ps[0])
    pt[1].data[:] = ps[1].data[:]
for ps, pt in zip(source_buffs['other'], target_buffs['other']):
    # print(pt[0], ps[0])
    pt[1].data[:] = ps[1].data[:]

print(len(source_params['fuse']), len(source_params['other']))
print(len(target_params['fuse']), len(target_params['other']))

# hmap branch 
source_params = []
target_params = []
source_buffs = []
target_buffs = []
for p in net_dekr.transition_hmap.named_parameters():
    target_params.append(p)
for p in net_dekr.transition_hmap.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    target_buffs.append(p)
for p in net_dekr.head_hmap.named_parameters():
    target_params.append(p)
for p in net_dekr.head_hmap.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    target_buffs.append(p)
for p in net_dekr.conv_hmap.named_parameters():
    target_params.append(p)
for p in net_dekr.conv_hmap.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    target_buffs.append(p)


for p in model.transition_heatmap.named_parameters():
    source_params.append(p)
for p in model.transition_heatmap.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    source_buffs.append(p)
for p in model.head_heatmap.named_parameters():
    source_params.append(p)
for p in model.head_heatmap.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    source_buffs.append(p)

for ps, pt in zip(source_params, target_params):
    # print(pt[0], ps[0])
    pt[1].data[:] = ps[1].data[:]
for ps, pt in zip(source_buffs, target_buffs):
    # print(pt[0], ps[0])
    pt[1].data[:] = ps[1].data[:]


print(len(source_params), len(source_buffs))
print(len(target_params), len(target_buffs))

# offset branch 
source_params = []
target_params = []
source_buffs = []
target_buffs = []
for p in net_dekr.transition_off.named_parameters():
    target_params.append(p)
for p in net_dekr.transition_off.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    target_buffs.append(p)
for p in net_dekr.reg_blks_off.named_parameters():
    target_params.append(p)
for p in net_dekr.reg_blks_off.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    target_buffs.append(p)
for p in net_dekr.convs_off.named_parameters():
    target_params.append(p)
for p in net_dekr.convs_off.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    target_buffs.append(p)


for p in model.transition_offset.named_parameters():
    source_params.append(p)
for p in model.transition_offset.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    source_buffs.append(p)
for p in model.offset_feature_layers.named_parameters():
    source_params.append(p)
for p in model.offset_feature_layers.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    source_buffs.append(p)
for p in model.offset_final_layer.named_parameters():
    source_params.append(p)
for p in model.offset_final_layer.named_buffers():
    name = p[0]
    if 'tracked' in name:
        continue
    source_buffs.append(p)

for ps, pt in zip(source_params, target_params):
    # print(pt[0], ps[0])
    pt[1].data[:] = ps[1].data[:]
for ps, pt in zip(source_buffs, target_buffs):
    # print(pt[0], ps[0])
    pt[1].data[:] = ps[1].data[:]

print(len(source_params), len(source_buffs))
print(len(target_params), len(target_buffs))

ylist = model.forward_test(x)
# print(ylist[0].shape, ylist[1].shape, ylist[2].shape, ylist[3].shape)
print(ylist[0], ylist[0].shape)
# print(ylist.shape)

# net.bn_eps(1e-5)
y2 = net_dekr(x)
print(y2[0], y2[0].shape)
print()
