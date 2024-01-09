# https://github.com/czczup/FAST
# https://arxiv.org/pdf/2111.02394.pdf


def infer(image_path: str, scale=1, target_format="png") -> str:
    pass


from FAST.models import FAST
import torch
from FAST.models.utils import fuse_module, rep_model_convert
from prepare_input import process_image

model = FAST()
model = model.cuda()
checkpoint = torch.load('model/weights.pth')
state_dict = checkpoint['ema']
d = dict()
for key, value in state_dict.items():
    tmp = key.replace("module.", "")
    d[tmp] = value

model.load_state_dict(d)
model = rep_model_convert(model)
model = fuse_module(model)
model.eval()
image = process_image("lp.jpg")
image['imgs'] = image['imgs'].cuda(non_blocking=True)

with torch.no_grad():
    outputs = model(**image)
    print(outputs)