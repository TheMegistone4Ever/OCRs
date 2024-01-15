import re
import easyocr
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2GRAY, IMWRITE_JPEG_QUALITY
import cv2
import os
import sys
import torch
from time import time
from multiprocessing import Pool, Queue, Process

from sys import getsizeof, stderr
from itertools import chain
from collections import deque

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


def job(input: Queue, output: Queue):
    while True:
        task = input.get()
        print("worker started")
        output.put(infer(task))
        print("worker finished")


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
    result = reader.readtext_batched(cv_images, detail=detail, batch_size=len(image_path_list), n_width=200,
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
    return infer([path])


def main():
    input_files = r"images/cropped/test"

    filenames = sorted(f for f in os.listdir(input_files) if os.path.isfile(os.path.join(input_files, f)) and f.endswith("png"))
    paths = [os.path.join(input_files, file) for file in filenames]
    start = time()
    num_processes = 10
    with Pool(num_processes) as pool:
        print("pool started")
        results = pool.map(infer_wrapper, paths)

    print(f"Time: {time() - start}")


if __name__ == "__main__":
    main()

    # start = time()
    # result = infer(paths)
    # print(f"Time: {time() - start}")
    # print("available: ", torch.cuda.current_device())

    # processes = []
    # for i in range(3):
    #     p = Process(target=job, args=(input,output))
    #     p.start()
    #     processes.append(p)

    # print("processed started", time() - start)
    # for i in range(6):
    #     input.put(path)

    # print
    # for p in processes:
    #     p.join()

    # for p in processes:
    #     p.terminate()

    # print(f"Time: {time() - start}")