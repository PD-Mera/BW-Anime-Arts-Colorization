
import os
import torch

from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from siggraph import SIGGRAPHGenerator
from config import TEST_CONFIG
from utils import get_norm_layer, tensor2im, rgb2lab, lab2rgb


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
if TEST_CONFIG["test_rgb"] is not None or TEST_CONFIG["test_rgb"] != "":
    rgb_img = Image.open(TEST_CONFIG["test_rgb"]).convert("RGB")
    ab_tensor = rgb2lab(ToTensor()(rgb_img).unsqueeze(0))[:,1:,:,:]
    gray_tensor = rgb2lab(ToTensor()(rgb_img).unsqueeze(0))[:,:1,:,:]
elif TEST_CONFIG["test_gray"] is not None or TEST_CONFIG["test_gray"] != "":
    gray_img = Image.open(TEST_CONFIG["test_rgb"]).convert("L")
    gray_tensor = rgb2lab(ToTensor()(gray_img).unsqueeze(0))
else:
    print("Config error. Exit...")
    exit()


# Get hint tensor
if TEST_CONFIG["hint_rgb"].lower() == "random":
    
    hint = Image.new(mode="RGB", size=(256, 256)).convert("LAB")
elif TEST_CONFIG["hint_rgb"] is not None or TEST_CONFIG["hint_rgb"] != "":
    hint = Image.open(TEST_CONFIG["hint_rgb"]).convert("LAB") 
else:
    hint = Image.new(mode="RGB", size=(256, 256)).convert("LAB")
hint_tensor = rgb2lab(ToTensor()(hint).unsqueeze(0))[:,1:,:,:]

# Get mask tensor
mask_tensor = torch.where(torch.sum(hint_tensor, dim=1) > 0.0, 1.0, 0.0).unsqueeze(1)
mask_tensor -= 0.5

# forward model
outputs = models(gray_tensor, hint_tensor, mask_tensor)

# save results
if TEST_CONFIG["test_rgb"] is not None or TEST_CONFIG["test_rgb"] != "":
    gray_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), torch.zeros_like(hint_tensor).type(torch.FloatTensor)), dim=1))
    real_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), ab_tensor.type(torch.FloatTensor)), dim=1))
    fake_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), outputs[1].type(torch.FloatTensor)), dim=1))
elif TEST_CONFIG["test_gray"] is not None or TEST_CONFIG["test_gray"] != "":
    gray_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), torch.zeros_like(hint_tensor).type(torch.FloatTensor)), dim=1))
    real_image = gray_image
    fake_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), outputs[1].type(torch.FloatTensor)), dim=1))


os.makedirs(TEST_CONFIG["test_dir"], exist_ok=True)
Image.fromarray(tensor2im(real_image)).save(os.path.join(TEST_CONFIG["test_dir"], "real.jpg"))
Image.fromarray(tensor2im(gray_image)).save(os.path.join(TEST_CONFIG["test_dir"], "gray.jpg"))
Image.fromarray(tensor2im(fake_image)).save(os.path.join(TEST_CONFIG["test_dir"], "fake.jpg"))