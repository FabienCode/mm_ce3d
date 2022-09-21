import torch

latest = torch.load("/data2/yaoming/project/mmdetection3d/log/eccv_r/pp/car_kitti_ori_v1/epoch_80.pth", map_location=torch.device('cpu'))
# latest = torch.load("/data/private/hym/project/fcaf3d_midea/log/log_eccv_r/3dssd/pc_kitti_4*4_v2/epoch_80.pth", map_location=torch.device('cpu'))
for k, v in list(latest['state_dict'].items()):
    if 'backbone' in k:
        tmp = k
        tmp_after = k.replace('label_backbone.', '')
        latest['state_dict'][tmp_after] = latest['state_dict'].pop(tmp)
    else:
        latest["state_dict"].pop(k)
    # if 'gt_backbone' in k or 'label_backbone' in k or 'pointnet' in k:
    #     latest["state_dict"].pop(k)
# for k, v in list(latest['state_dict'].items()):
#     if 'label' in k:
#         tmp = k
#         tmp_after = k.replace('label_backbone.', '')
#         latest['state_dict'][tmp_after] = latest['state_dict'].pop(tmp)
#     else:
#         latest["state_dict"].pop(k)
torch.save(latest, '/data2/yaoming/project/mmdetection3d/data/pp_kitti_labelbackbone.pth')
print("hahahah")