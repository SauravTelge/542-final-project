import torch

from torch.nn.functional import interpolate

def resize_depth(depth, target_h, target_w):
    origin_h, origin_w = depth.shape[2:]
    if target_h == origin_h and target_w == origin_w:
        return depth
    depth = interpolate(depth, size=(target_h, target_w), mode='bilinear', align_corners=True)
    # depth[:, 0] *= target_w / origin_w
    # depth[:, 1] *= target_h / origin_h
    return depth

def si_log_loss(pred, gt):
    diff = torch.log(pred) - torch.log(gt)
    n = pred.shape[-1] * pred.shape[-2]
    sq_diff = diff ** 2
    loss = (sq_diff.sum() / n) - ((diff.sum() ** 2) / (n ** 2))
    return loss


def get_depth_loss(depths, targets, type='robust'):

    total_loss = 0
    diff = torch.abs(resize_depth(depths, targets.shape[2], targets.shape[3]) - targets)
    # diff = torch.abs(depths - targets)

    if type == 'robust':
        loss_l1_map = torch.pow(diff.sum(dim=1) + 0.01, 0.4)
        loss_l1 = loss_l1_map.mean()
        total_loss = loss_l1/depths.shape[0]
    elif type == 'mse':
       
        loss_l2 = torch.nn.functional.mse_loss(resize_depth(depths, targets.shape[2], targets.shape[3]), targets)
        total_loss = loss_l2/depths.shape[0]
    else:
        raise NotImplementedError

    return total_loss


