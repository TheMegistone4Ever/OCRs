# https://github.com/czczup/FAST
# https://arxiv.org/pdf/2111.02394.pdf


def infer(image_path: str, scale=1) -> str:
    pass


# from model import FAST
# import torch
# from model.utils import fuse_module, rep_model_convert
# from prepare_input import process_image
#
# model = FAST()
# model = model.cuda()
# checkpoint = torch.load('model/weights.pth')
# state_dict = checkpoint['ema']
# d = dict()
# for key, value in state_dict.items():
#     tmp = key.replace("module.", "")
#     d[tmp] = value
#
# model.load_state_dict(d)
# model = rep_model_convert(model)
# model = fuse_module(model)
# model.eval()
#
# import numpy as np
#
# image = process_image(r"images\lp.jpg")
# image['imgs']= image['imgs'].cuda(non_blocking=True)
#
# if __name__ == '__main__':
#     with torch.no_grad():
#         outputs = model(**image)
#         print(outputs)
