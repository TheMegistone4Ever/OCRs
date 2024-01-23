import math
import re
import easyocr
import numpy as np
from PIL import JpegImagePlugin
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2GRAY, IMWRITE_JPEG_QUALITY
import cv2
import os
import sys
import torch
from time import time, sleep
from multiprocessing import Pool, Queue, Process
from skimage.io._io import imread
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
import matplotlib.pyplot as plt
from FifaReader import FifaReader
from InputItem import InputItem

fifa_reader = FifaReader(["en"])

try:
    from reprlib import repr
except ImportError:
    pass


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


reader = easyocr.Reader(["en"], gpu=True)


# reader = easyocr.Reader(["en"], gpu=True, detect_network="dbnet18")

def job(input: Queue, output: Queue):
    while True:
        task = input.get()
        print("worker started")
        output.put(infer(task))
        print("worker finished")


def loadImage(img_file):
    img = imread(img_file)  # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:, :, :3]
    img = np.array(img)

    return img


def reformat_input(image):
    if type(image) == str:
        img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image = os.path.expanduser(image)
        img = loadImage(image)  # can accept URL

    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            img_cv_grey = image
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img_cv_grey = np.squeeze(image)
            img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # BGRscale
            img = image
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
            img = image[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif type(image) == JpegImagePlugin.JpegImageFile:
        image_array = np.array(image)
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Invalid input type. Supporting format = string(file path or url), bytes, numpy array')

    return img, img_cv_grey


def readtext(image, reader, horizontal_list, free_list, with_detection,
             decoder='greedy', beamWidth=5, batch_size=8, workers=0, allowlist=None,
             blocklist=None, detail=1,
             rotation_info=None, paragraph=False, min_size=20, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003,
             text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1., slope_ths=0.1,
             ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1,
             threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, output_format='standard'):
    """
    Parameters:
    image: file path or numpy-array or a byte stream object
    """

    img, img_cv_grey = reformat_input(image)

    if with_detection:
        horizontal_list, free_list = reader.detect(img,
                                                   min_size=min_size, text_threshold=text_threshold,
                                                   low_text=low_text, link_threshold=link_threshold,
                                                   canvas_size=canvas_size, mag_ratio=mag_ratio,
                                                   slope_ths=slope_ths, ycenter_ths=ycenter_ths,
                                                   height_ths=height_ths, width_ths=width_ths,
                                                   add_margin=add_margin, reformat=False,
                                                   threshold=threshold, bbox_min_score=bbox_min_score,
                                                   bbox_min_size=bbox_min_size, max_candidates=max_candidates
                                                   )

        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        horizontal_list, free_list = horizontal_list[0], free_list[0]

    result = reader.recognize(img_cv_grey, horizontal_list, free_list,
                              decoder, beamWidth, batch_size,
                              workers, allowlist, blocklist, detail, rotation_info,
                              paragraph, contrast_ths, adjust_contrast,
                              filter_ths, y_ths, x_ths, False, output_format)

    return result


def reformat_input_batched(image, n_width=None, n_height=None):
    """
    reformats an image or list of images or a 4D numpy image array &
    returns a list of corresponding img, img_cv_grey nd.arrays
    image:
        [file path, numpy-array, byte stream object,
        list of file paths, list of numpy-array, 4D numpy array,
        list of byte stream objects]
    """
    if (isinstance(image, np.ndarray) and len(image.shape) == 4) or isinstance(image, list):
        # process image batches if image is list of image np arr, paths, bytes
        img, img_cv_grey = [], []
        for single_img in image:
            clr, gry = reformat_input(single_img)
            if n_width is not None and n_height is not None:
                clr = cv2.resize(clr, (n_width, n_height))
                gry = cv2.resize(gry, (n_width, n_height))
            img.append(clr)
            img_cv_grey.append(gry)
        img, img_cv_grey = np.array(img), np.array(img_cv_grey)
        # ragged tensors created when all input imgs are not of the same size
        if len(img.shape) == 1 and len(img_cv_grey.shape) == 1:
            raise ValueError("The input image array contains images of different sizes. " +
                             "Please resize all images to same shape or pass n_width, n_height to auto-resize")
    else:
        img, img_cv_grey = reformat_input(image)
    return img, img_cv_grey


def readtext_batched(image, n_width=None, n_height=None, \
                     decoder='greedy', beamWidth=5, batch_size=1, \
                     workers=0, allowlist=None, blocklist=None, detail=1, \
                     rotation_info=None, paragraph=False, min_size=20, \
                     contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, \
                     text_threshold=0.7, low_text=0.4, link_threshold=0.4, \
                     canvas_size=2560, mag_ratio=1., \
                     slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, \
                     width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1,
                     threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0,
                     output_format='standard'):
    """
    Parameters:
    image: file path or numpy-array or a byte stream object
    When sending a list of images, they all most of the same size,
    the following parameters will automatically resize if they are not None
    n_width: int, new width
    n_height: int, new height
    """
    img, img_cv_grey = reformat_input_batched(image, n_width, n_height)

    # horizontal_list_agg, free_list_agg = reader.detect(img,
    #                                                  min_size=min_size, text_threshold=text_threshold,
    #                                                  low_text=low_text, link_threshold=link_threshold,
    #                                                  canvas_size=canvas_size, mag_ratio=mag_ratio, slope_ths=slope_ths, ycenter_ths=ycenter_ths,
    #                                                  height_ths=height_ths, width_ths=width_ths, add_margin=add_margin, reformat=False,
    #                                                  threshold=threshold, bbox_min_score=bbox_min_score,
    #                                                  bbox_min_size=bbox_min_size, max_candidates=max_candidates
    #                                                  )

    horizontal_list_agg = [[0, 0, n_width - 1, n_height - 1]] * len(img)
    free_list_agg = [[]] * len(img)

    result_agg = reader.recognize(img_cv_grey, horizontal_list_agg, free_list_agg,
                                  decoder, beamWidth, batch_size,
                                  workers, allowlist, blocklist, detail, rotation_info,
                                  paragraph, contrast_ths, adjust_contrast,
                                  filter_ths, y_ths, x_ths, False, output_format)
    return result_agg


def infer(image_path_list: str, scale=1, detail=0) -> str:
    """
    Infer text from image using EasyOCR.
    :param image_path: Path to an image
    :param scale: Scale factor for image
    :param detail: 0 - low, 1 - medium, 2 - high
    :param target_format: Target format for image
    :param reader: EasyOCR reader
    :return: text
    """

    cv_image = cv2.imread(image_path_list)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    return re.sub(r"\s+", " ", " ".join(reader.readtext(cv_image, detail=detail, batch_size=16))).strip().lower()
    print(f"NIKITA LIST: {image_path_list}")
    cv_images = [cvtColor(imread(image_path), COLOR_BGR2GRAY) for image_path in image_path_list]
    result = reader.readtext_batched(cv_images, detail=detail, batch_size=len(image_path_list), n_width=180,
                                     n_height=200)
    return result


# if __name__ == "__main__":
#
#     filenames = sorted(f for f in os.listdir(input_files) if os.path.isfile(os.path.join(input_files, f)))
#     paths = [os.path.join(input_files, file) for file in filenames]
#
#     start = time()
#     for path in paths:
#         result = infer([path])
#     print(f"Time: {time() - start}")


def infer_wrapper(path):
    return infer(path)


def find_bbox(cv_image, text_color=0, padding=0):
    h = cv_image.shape[0]
    w = cv_image.shape[1]

    min_x_zero, min_y_zero = w - 1, h - 1
    max_x_zero, max_y_zero = 0, 0
    counter_zero = 0

    min_x_nonzero, min_y_nonzero = w - 1, h - 1
    max_x_nonzero, max_y_nonzero = 0, 0

    for y in range(0, h):
        for x in range(0, w):
            if cv_image[y, x] == text_color:
                min_x_zero = min(min_x_zero, x)
                min_y_zero = min(min_y_zero, y)
                max_x_zero = max(max_x_zero, x)
                max_y_zero = max(max_y_zero, y)
                counter_zero += 1
            else:
                min_x_nonzero = min(min_x_nonzero, x)
                min_y_nonzero = min(min_y_nonzero, y)
                max_x_nonzero = max(max_x_nonzero, x)
                max_y_nonzero = max(max_y_nonzero, y)

    if counter_zero > h * w / 2 + w:
        return min_x_nonzero, min_y_nonzero, max_x_nonzero, max_y_nonzero

    return min_x_zero, min_y_zero, max_x_zero, max_y_zero


def main(a=0):
    input_file = r"images/cropped/test/Controls.png"
    # cv_image = cv2.imread(input_file)

    # images = [os.path.join(input_file, file) for file in os.listdir(input_file)]
    images = [input_file] * 100
    # decoders = ["greedy", "beamsearch", "wordbeamsearch"]

    decoders = ["wordbeamsearch"]

    for decoder in decoders:
        all_time = 0
        # if decoder != "wordbeamsearch":
        #     continue

        fig, axs = plt.subplots(len(images), 3, figsize=(6, len(images) * .7), dpi=200)
        # fig, axs = plt.subplots(1, 3, figsize=(6, 2), dpi=200)
        fig.suptitle(f"Decoder: {decoder}")

        fig.patch.set_facecolor('#cccccc')

        k = -1
        for i, image_path in enumerate(images):
            # if k < 10:
            #     k += 1
            #     continue
            #
            # if k > 10:
            #     break
            k += 1

            cv_image = cv2.imread(image_path)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # cv_image = cv2.resize(cv_image, (cv_image.shape[1] * 4, cv_image.shape[0] * 4))

            # full contrast
            # cv_image = cv2.equalizeHist(cv_image)

            # remove noise
            # cv_image = cv2.fastNlMeansDenoising(cv_image, h=10, templateWindowSize=5, searchWindowSize=11)

            # Apply binary thresholding
            # _, cv_image = cv2.threshold(cv_image, 128, 255, cv2.THRESH_BINARY)

            # _, cv_image = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # cv_image = cv2.copyMakeBorder(cv_image, 8, 8, 8, 8, cv2.BORDER_CONSTANT)

            # cv_image = cv2.bitwise_not(cv_image)

            # cv2.imshow("image", cv_image)
            # cv2.waitKey(0)

            # cv2.imwrite("test.bmp", cv_image)

            free_list = []

            # horizontal_list = [
            #     [0, cv_image.shape[1] - 1, 0, cv_image.shape[0] - 1]
            # ]

            # readtext(image, reader, horizontal_list, free_list, with_detection,
            #          decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None,
            #          blocklist=None, detail=1,
            #          rotation_info=None, paragraph=False, min_size=20, contrast_ths=0.1, adjust_contrast=0.5,
            #          filter_ths=0.003,
            #          text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1.,
            #          slope_ths=0.1,
            #          ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1,
            #          threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, output_format='standard')

            top_left_px = cv_image[0, 0]
            top_right_px = cv_image[0, cv_image.shape[1] - 1]
            bottom_left_px = cv_image[cv_image.shape[0] - 1, 0]
            bottom_right_px = cv_image[cv_image.shape[0] - 1, cv_image.shape[1] - 1]

            # counter = dict()
            #
            # for px in [top_left_px, top_right_px, bottom_left_px, bottom_right_px]:
            #     if px not in counter:
            #         counter[px] = 0
            #     counter[px] += 1
            #
            # background = max(counter, key=counter.get)

            # background = 255
            # if top_left_px == bottom_right_px:
            #     if top_right_px == top_left_px or bottom_left_px == top_left_px:
            #         background = top_left_px
            #     else:
            #         pass
            # else:
            #     if top_right_px == bottom_left_px:
            #         background = top_right_px
            #     else:
            #         pass

            border = 4
            # min_x, min_y, max_x, max_y = find_bbox(cv_image, background)

            # full table
            # cv_image = cv_image[105:572, 373:1175]

            # one colum
            # cv_image = cv_image[120:555, 522:558]

            scale = 1
            # cv_image = cv2.resize(cv_image, (int(cv_image.shape[1] * scale), int(cv_image.shape[0] * scale)))
            min_x, min_y, max_x, max_y = 0, 0, cv_image.shape[1] - 1, cv_image.shape[0] - 1

            min_x = max(0, min_x - border)
            min_y = max(0, min_y - border)
            max_x = min(cv_image.shape[1] - 1, max_x + border)
            max_y = min(cv_image.shape[0] - 1, max_y + border)

            horizontal_list = [
                [min_x, max_x, min_y, max_y]
            ]

            def nearest_power_of_2(x: int) -> int:
                return 2 ** round(math.log2(x))

            def get_best_batch_size(image_size: int) -> int:
                x = max(4, int(96 * math.log(0.0005 * image_size)))
                return nearest_power_of_2(x)

            best_batch_size = get_best_batch_size(cv_image.shape[0] * cv_image.shape[1] // 3)
            input_item = InputItem.from_path(image_path, scale=scale)

            if k % 1 == 0:
                sleep(.2)
                pass

            start = time()

            # result = readtext(cv_image, reader, horizontal_list, free_list, False, decoder=decoder,
            #                   batch_size=best_batch_size)

            result = fifa_reader.read(input_item)

            eval_time = time() - start

            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # for res in result:
            #     box = res[0]
            #     pt1 = box[0]
            #     pt2 = box[2]
            #
            #     pt1 = int(pt1[0]), int(pt1[1])
            #     pt2 = int(pt2[0]), int(pt2[1])
            #
            #     cv_image = cv2.rectangle(cv_image, pt1, pt2, (0, 0, 255), 2)
            #
            # cv2.imshow("image", cv_image)
            # cv2.waitKey(0)

            if i != 0:
                all_time += eval_time

            if len(result.result) > 0:
                result_text = len(result.result)



                # box = result[0][0]
                # pt1 = box[0]
                # pt2 = box[2]
                # pt1 = int(pt1[0]), int(pt1[1])
                # pt2 = int(pt2[0]), int(pt2[1])
                #
                # cv_image = cv2.rectangle(cv_image, pt1, pt2, (255, 0, 0), 1)
                #
                # # Display the text result
                # result_text = "\n".join([res[1] for res in result])
                axs[i, 1].text(0.5, 0.5, f"Text: {result_text}", ha='center', va='center', wrap=True)

            axs[i, 1].axis('off')

            # Display the evaluation time
            axs[i, 2].text(0.5, 0.5, f"Time: {eval_time:.5f} seconds", ha='center', va='center')
            axs[i, 2].axis('off')

            # Display the image
            axs[i, 0].imshow(cv_image)
            axs[i, 0].axis('off')

        plt.show()

        print(f"{decoder = }")
        print(f"{all_time = }")
        print(f"{all_time / (len(images) - 1) = }")
        print()


def main_benchmark():
    input_file = r"images/tesseract_errors2"
    images = [
        cv2.imread(os.path.join(input_file, file))
        for file in os.listdir(input_file)]
    decoders = ["greedy", "beamsearch", "wordbeamsearch"]
    decoder = "wordbeamsearch"

    result = readtext(images, decoder=decoder, batch_size=16, n_width=200, n_height=200)
    print(f"{result = }")


from multiprocessing import Pool

if __name__ == '__main__':
    main()
