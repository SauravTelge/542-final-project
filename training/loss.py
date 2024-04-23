import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def mse_loss(input_flow, target_flow):

    # output_transform = transforms.Compose(
    #     [
    #         # transforms.ToTensor(),
    #         transforms.Normalize(mean=[0, 0], std=[20, 20]),
    #     ]
    # )

    loss = torch.nn.MSELoss()
    b, _, h, w = target_flow.size()
    upsampled_output = F.interpolate(
        input_flow, (h, w), mode="bilinear", align_corners=False
    )

    # upsampled_output = output_transform(upsampled_output)
    return torch.sqrt(loss(upsampled_output, target_flow))

def EPE(input_flow, target_flow, sparse=False, mean=True):

    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size



def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(
        output, (h, w), mode="bilinear", align_corners=False
    )
    return EPE(upsampled_output, target, sparse, mean=True)