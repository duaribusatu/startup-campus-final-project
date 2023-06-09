import torch
from RealESRGAN import RealESRGAN


def init_model(model_path , device):
    inf_model = torch.load(model_path)
    inf_model.to(device)
    inf_model.eval()

    return inf_model

def init_supres(weight_path = 'weight/RealESRGAN_x4plus.pth', device = 'cpu'):
    device = torch.device('cpu')
    supres = RealESRGAN(device, scale=4)
    supres.load_weights(weight_path)

    return supres