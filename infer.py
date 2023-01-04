
import os
import torch

from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from siggraph import SIGGRAPHGenerator
from config import TEST_CONFIG
from utils import get_norm_layer, tensor2im, rgb2lab, lab2rgb, decode_max_ab, decode_mean


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
    rgb_img = Image.open(TEST_CONFIG["test_rgb"]).convert("RGB")
    ab_tensor = rgb2lab(ToTensor()(rgb_img).unsqueeze(0))[:,1:,:,:]
    gray_tensor = rgb2lab(ToTensor()(rgb_img).unsqueeze(0))[:,:1,:,:]
elif TEST_CONFIG["test_gray"] is not None and TEST_CONFIG["test_gray"] != "":
    gray_img = Image.open(TEST_CONFIG["test_rgb"]).convert("L")
    gray_tensor = rgb2lab(ToTensor()(gray_img).unsqueeze(0))
else:
    print("Config error. Exit...")
    exit()


# Get hint tensor
if TEST_CONFIG["hint"] == None:
    hint = Image.new(mode="RGB", size=(256, 256))
    hint_tensor = rgb2lab(ToTensor()(hint).unsqueeze(0))[:,1:,:,:]

elif TEST_CONFIG["hint"] == 'local':
    hint = Image.open(TEST_CONFIG["hint_rgb"])
    hint_tensor = rgb2lab(ToTensor()(hint).unsqueeze(0))[:,1:,:,:]

elif TEST_CONFIG["hint"] == 'global':
    pass
    # hint = Image.open(TEST_CONFIG["hint_rgb"])
    # gray_zero = rgb2lab(ToTensor()(hint).unsqueeze(0))[:,:1,:,:]
    # hint_zero = rgb2lab(ToTensor()(hint).unsqueeze(0))[:,1:,:,:]
    # mask_zero = torch.where(torch.sum(hint_zero, dim=1) > 0.0, 1.0, 0.0).unsqueeze(1)
    # mask_zero -= 0.5
    # fake_class, _ = models(gray_zero, hint_zero, mask_zero)
    # fake_dec_max = models.upsample4(decode_max_ab(fake_class))
    # fake_distr = models.softmax(fake_class)
    # fake_dec_mean = models.upsample4(decode_mean(fake_distr))
    # # fake_entr = models.upsample4(-torch.sum(fake_distr * torch.log(fake_distr + 1.e-10), dim=1, keepdim=True))
    # fake_entr = models.upsample4(fake_distr)
    # print(fake_dec_mean.size())
    # hint_tensor = fake_distr
    # print(hint_tensor)

else:
    print("Config error. Exit...")
    exit()




# Get mask tensor
mask_tensor = torch.where(torch.sum(hint_tensor, dim=1) > 0.0, 1.0, 0.0).unsqueeze(1)
mask_tensor -= 0.5

# forward model
_, fake_reg = models(gray_tensor, hint_tensor, mask_tensor)

# exit()

# save results
if TEST_CONFIG["test_rgb"] is not None and TEST_CONFIG["test_rgb"] != "":
    gray_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), torch.zeros_like(hint_tensor).type(torch.FloatTensor)), dim=1))
    real_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), ab_tensor.type(torch.FloatTensor)), dim=1))
    fake_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), fake_reg.type(torch.FloatTensor)), dim=1))
elif TEST_CONFIG["test_gray"] is not None and TEST_CONFIG["test_gray"] != "":
    gray_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), torch.zeros_like(hint_tensor).type(torch.FloatTensor)), dim=1))
    real_image = gray_image
    fake_image = lab2rgb(torch.cat((gray_tensor.type(torch.FloatTensor), fake_reg.type(torch.FloatTensor)), dim=1))


os.makedirs(TEST_CONFIG["test_dir"], exist_ok=True)
Image.fromarray(tensor2im(real_image)).save(os.path.join(TEST_CONFIG["test_dir"], "real.jpg"))
Image.fromarray(tensor2im(gray_image)).save(os.path.join(TEST_CONFIG["test_dir"], "gray.jpg"))
Image.fromarray(tensor2im(fake_image)).save(os.path.join(TEST_CONFIG["test_dir"], "fake.jpg"))