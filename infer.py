
import os
import torch

from PIL import Image
from torchvision.transforms import ToTensor

from siggraph import SIGGRAPHGenerator
from config import TEST_CONFIG
from utils import get_norm_layer, tensor2im, rgb2lab, lab2rgb, rgb2hsv, pil_loader
from model_util import encode_global


input_num_channels = 1
output_num_channels = 2
input_num_channels = input_num_channels + output_num_channels + 1
norm_layer = get_norm_layer(norm_type='batch')
use_tanh = True
classification = False

models = SIGGRAPHGenerator(input_num_channels, output_num_channels, norm_layer=norm_layer, use_tanh=use_tanh, classification=classification)
models.load_state_dict(torch.load(TEST_CONFIG["pretrained"]))
models.eval()

# Get gray tensor
if TEST_CONFIG["test_rgb"] is not None and TEST_CONFIG["test_rgb"] != "":
    rgb_img = pil_loader(TEST_CONFIG["test_rgb"]).convert("RGB")
    ab_tensor = rgb2lab(ToTensor()(rgb_img).unsqueeze(0))[:,1:,:,:]
    gray_tensor = rgb2lab(ToTensor()(rgb_img).unsqueeze(0))[:,:1,:,:]
elif TEST_CONFIG["test_gray"] is not None and TEST_CONFIG["test_gray"] != "":
    gray_img = Image.open(TEST_CONFIG["test_rgb"]).convert("RGB")
    gray_tensor = rgb2lab(ToTensor()(gray_img).unsqueeze(0))[:,:1,:,:]
else:
    print("Config error. Exit...")
    exit()


# Get hint tensor
if TEST_CONFIG["hint"] == None:
    # Hint local 
    dummy_hint = Image.new(mode="RGB", size=(256, 256))
    hint_local_tensor = rgb2lab(ToTensor()(dummy_hint).unsqueeze(0))[:,1:,:,:]

    # Mask local
    mask_local_tensor = torch.where(torch.sum(hint_local_tensor, dim=1) > 0.0, 1.0, 0.0).unsqueeze(1)
    mask_local_tensor -= 0.5

    # Hint global
    ab_global_tensor = rgb2lab(ToTensor()(dummy_hint).unsqueeze(0))[:,1:,:,:]
    hint_global_tensor = encode_global(ab_global_tensor).unsqueeze(2).unsqueeze(3)

    # Mask global
    mask_global_tensor = torch.zeros((1, 1, 1, 1))

    # Hint s
    hint_s_tensor = rgb2hsv(ToTensor()(dummy_hint).unsqueeze(0))[:,1:2,:,:]
    hint_s_tensor = torch.mean(hint_s_tensor, dim=3)
    hint_s_tensor = torch.mean(hint_s_tensor, dim=2).unsqueeze(2).unsqueeze(3)

    # Mask s
    mask_s_tensor = torch.zeros((1, 1, 1, 1))
    

elif TEST_CONFIG["hint"] == 'local':
    # Hint local 
    local_hint = pil_loader(TEST_CONFIG["hint_rgb"])
    hint_local_tensor = rgb2lab(ToTensor()(local_hint).unsqueeze(0))[:,1:,:,:]

    # Mask local
    mask_local_tensor = torch.where(torch.sum(hint_local_tensor, dim=1) > 0.0, 1.0, 0.0).unsqueeze(1)
    mask_local_tensor -= 0.5

    # Hint global
    dummy_hint = Image.new(mode="RGB", size=(256, 256))
    ab_global_tensor = rgb2lab(ToTensor()(dummy_hint).unsqueeze(0))[:,1:,:,:]
    hint_global_tensor = encode_global(ab_global_tensor).unsqueeze(2).unsqueeze(3)
    hint_global_tensor = torch.ones((1, 313, 1, 1)) / 313

    # Mask global
    mask_global_tensor = torch.zeros((1, 1, 1, 1))

    # Hint s
    hint_s_tensor = rgb2hsv(ToTensor()(dummy_hint).unsqueeze(0))[:,1:2,:,:]
    hint_s_tensor = torch.mean(hint_s_tensor, dim=3)
    hint_s_tensor = torch.mean(hint_s_tensor, dim=2).unsqueeze(2).unsqueeze(3)

    # Mask s
    mask_s_tensor = torch.zeros((1, 1, 1, 1))


elif TEST_CONFIG["hint"] == 'global':
    # Hint local 
    dummy_hint = Image.new(mode="RGB", size=(256, 256))
    hint_local_tensor = rgb2lab(ToTensor()(dummy_hint).unsqueeze(0))[:,1:,:,:]

    # Mask local
    mask_local_tensor = torch.where(torch.sum(hint_local_tensor, dim=1) > 0.0, 1.0, 0.0).unsqueeze(1)
    mask_local_tensor -= 0.5

    # Hint global
    global_hint = pil_loader(TEST_CONFIG["hint_rgb"])
    ab_global_tensor = rgb2lab(ToTensor()(global_hint).unsqueeze(0))[:,1:,:,:]
    hint_global_tensor = encode_global(ab_global_tensor).unsqueeze(2).unsqueeze(3)
    # hint_global_tensor = torch.ones((1, 313, 1, 1)) / 313


    # Mask global
    mask_global_tensor = torch.ones((1, 1, 1, 1))

    # Hint s
    hint_s_tensor = rgb2hsv(ToTensor()(global_hint).unsqueeze(0))[:,1:2,:,:]
    hint_s_tensor = torch.mean(hint_s_tensor, dim=3)
    hint_s_tensor = torch.mean(hint_s_tensor, dim=2).unsqueeze(2).unsqueeze(3)

    # Mask s
    mask_s_tensor = torch.ones((1, 1, 1, 1))


else:
    print("Config error. Exit...")
    exit()


# forward model
_, fake_reg = models(gray_tensor, hint_local_tensor, mask_local_tensor,
                                  hint_global_tensor, mask_global_tensor,
                                  hint_s_tensor, mask_s_tensor)

# save results
if TEST_CONFIG["test_rgb"] is not None and TEST_CONFIG["test_rgb"] != "":
    gray_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), torch.zeros_like(hint_local_tensor).type(torch.FloatTensor)), dim=1))
    real_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), ab_tensor.type(torch.FloatTensor)), dim=1))
    fake_ab = lab2rgb(torch.cat((torch.zeros_like(gray_tensor), fake_reg.type(torch.FloatTensor)), dim=1))
    fake_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), fake_reg.type(torch.FloatTensor)), dim=1))
elif TEST_CONFIG["test_gray"] is not None and TEST_CONFIG["test_gray"] != "":
    gray_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), torch.zeros_like(hint_local_tensor).type(torch.FloatTensor)), dim=1))
    real_image = gray_image
    fake_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), fake_reg.type(torch.FloatTensor)), dim=1))


os.makedirs(TEST_CONFIG["test_dir"], exist_ok=True)
Image.fromarray(tensor2im(real_image)).save(os.path.join(TEST_CONFIG["test_dir"], "real.jpg"))
Image.fromarray(tensor2im(gray_image)).save(os.path.join(TEST_CONFIG["test_dir"], "gray.jpg"))
Image.fromarray(tensor2im(fake_image)).save(os.path.join(TEST_CONFIG["test_dir"], "fake.jpg"))
# Image.fromarray(tensor2im(fake_ab)).save(os.path.join(TEST_CONFIG["test_dir"], "fake_ab.jpg"))