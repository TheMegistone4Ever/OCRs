import torch
from easyocr.craft import CRAFT
from importlib import import_module
from os.path import join as path_join, expanduser
from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def pth_to_pt(path_to_pth, path_to_pt, net, device, is_detector, quantize=True):
    try:
        if is_detector:
            if device == 'cpu':
                net.load_state_dict(copyStateDict(torch.load(path_to_pth, map_location=device)))
                if quantize:
                    try:
                        torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
                    except Exception as e:
                        raise e
            else:
                net.load_state_dict(copyStateDict(torch.load(path_to_pth, map_location=device)))
                net = torch.nn.DataParallel(net).to(device)

            # net.eval()
            # traced_script_module = torch.jit.script(net)
            # traced_script_module.save(path_to_pt)
        else:
            if device == 'cpu':
                state_dict = torch.load(path_to_pth, map_location=device)
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    new_key = key[7:]
                    new_state_dict[new_key] = value
                net.load_state_dict(new_state_dict)
                if quantize:
                    try:
                        torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
                    except Exception as e:
                        raise e
            else:
                net = torch.nn.DataParallel(net).to(device)
                net.load_state_dict(torch.load(path_to_pth, map_location=device))
        net.eval()
        torch.save(net, path_to_pt)

    except Exception as e:
        raise e
    else:
        print(f"Conversion from \"{path_to_pth}\" to \"{path_to_pt}\" is done...")


if __name__ == "__main__":
    if torch.cuda.is_available():
        current_device = "cuda"
    elif torch.backends.mps.is_available():
        current_device = "mps"
    else:
        current_device = "cpu"

    current_device = "cpu"

    MODEL_PATH = expanduser(r"~\.EasyOCR\model")
    craft_model_path = path_join(MODEL_PATH, "craft_mlt_25k.pth")
    craft_pt_path = path_join(MODEL_PATH, "craft_mlt_25k.pt")
    craft_model_instance = CRAFT()
    pth_to_pt(craft_model_path, craft_pt_path, craft_model_instance, current_device, True)

    english_model_path = path_join(MODEL_PATH, "english_g2.pth")
    english_pt_path = path_join(MODEL_PATH, "english_g2.pt")
    recognition_network = "generation2"
    if recognition_network == "generation1":
        model_pkg = import_module("easyocr.model.model")
    elif recognition_network == "generation2":
        model_pkg = import_module("easyocr.model.vgg_model")
    else:
        model_pkg = import_module(recognition_network)
    characters = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    num_class = len(characters) + 1  # +1 for blank token
    network_params = {"hidden_size": 256, "input_channel": 1, "output_channel": 256}
    english_model_instance = model_pkg.Model(num_class=num_class, **network_params)
    pth_to_pt(english_model_path, english_pt_path, english_model_instance, current_device, False)
