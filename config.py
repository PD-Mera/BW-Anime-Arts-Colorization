TEST_CONFIG = {
    "pretrained": "./checkpoints/siggraph_reg2/4_net_G.pth",
    "test_rgb": "./assets/imgs/test_img.jpg", # Convert to gray and compare with results
                                               # If you set test_rgb, ignore test_gray
    "test_gray": "", # Function only when you set test_rgb = None
    "hint": "global", # None or "local" or "global" (global is currently unavailable)
    "hint_rgb": "./assets/imgs/global/hint.jpg", #"./assets/hint.png", # rgb image link
    "test_dir": "./sample",
}