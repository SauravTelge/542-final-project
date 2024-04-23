import torch
from torch.nn.functional import interpolate
import torch.nn.functional as F

def resize_flow2d(flow, target_h, target_w):
    origin_h, origin_w = flow.shape[2:]
    if target_h == origin_h and target_w == origin_w:
        return flow
    flow = interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=True)
    flow[:, 0] *= target_w / origin_w
    flow[:, 1] *= target_h / origin_h
    return flow

def EPE(input_flow, target_flow, sparse=False, mean=True):

    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size

def mse_loss(input_flow, target_flow):

    loss = torch.nn.MSELoss()
    b, _, h, w = target_flow.size()
    upsampled_output = F.interpolate(
        input_flow, (h, w), mode="bilinear", align_corners=False
    )

    return torch.sqrt(loss(upsampled_output, target_flow))

def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(
        output, (h, w), mode="bilinear", align_corners=False
    )
    return EPE(upsampled_output, target, sparse, mean=True)

def get_flow_loss(flows, targets, type='robust'):

    total_loss = 0
    
    assert flows.shape[1] == 2  # [B, 2, H, W]

    if targets.shape[1] == 3:
        flow_masks = targets[:, 2] > 0
    else:
        flow_masks = torch.ones_like(targets)[:, 0] > 0
    
    diff = torch.abs(resize_flow2d(flows, targets.shape[2], targets.shape[3]) - targets[:, :2])

    if type == 'robust':
        loss_l1_map = torch.pow(diff.sum(dim=1) + 0.01, 0.4)
        loss_l1 = loss_l1_map[flow_masks].mean()
        total_loss = loss_l1/flows.shape[0]
    elif type == 'mse':
        loss_l2 = torch.nn.functional.mse_loss(resize_flow2d(flows, targets.shape[2], targets.shape[3]), targets[:, :2])
        total_loss = loss_l2/flows.shape[0]
    else:
        raise NotImplementedError

    return total_loss


