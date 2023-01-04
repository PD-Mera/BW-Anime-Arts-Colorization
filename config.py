TEST_CONFIG = {
    "pretrained": "./checkpoints/siggraph_reg2/latest_net_G.pth",
    "test_rgb": "./assets/test_img.jpg", # Convert to gray and compare with results
                                               # If you set test_rgb, ignore test_gray
    "test_gray": "", # Function only when you set test_rgb = None
    "hint_rgb": "./assets/hint.png", # rgb image link or "random" or None (Now random is not support (= None))
    "test_dir": "./sample",
}