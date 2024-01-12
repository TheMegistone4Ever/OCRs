# https://github.com/JaidedAI/EasyOCR
# https://arxiv.org/pdf/1507.05717.pdf
# https://arxiv.org/pdf/1904.01941.pdf


import re
import easyocr
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2GRAY, IMWRITE_JPEG_QUALITY
import cv2
import os
from time import time


reader = easyocr.Reader(["en"], gpu=True)
def infer(image_path: str, scale=1, detail=0) -> str:
    """
    Infer text from image using EasyOCR.
    :param image_path: Path to an image
    :param scale: Scale factor for image
    :param detail: 0 - low, 1 - medium, 2 - high
    :param target_format: Target format for image
    :param reader: EasyOCR reader
    :return: text
    """
    cv_image = imread(image_path)
    cv_image = cvtColor(cv_image, COLOR_BGR2GRAY)

    # if not image_path.endswith(target_format):
    #     imwrite('image.jpg', cv_image, [int(IMWRITE_JPEG_QUALITY), 100])

    # if scale != 1:
    #     cv_image = cv_image.resize(cv_image.shape[0] * scale, cv_image.shape[1] * scale)

    return re.sub(r"\s+", " ", " ".join(reader.readtext(cv_image, detail=detail, batch_size=16))).strip().lower()

    # [[horizontal_list], _] = reader.detect(cv_image, min_size=1, canvas_size=2560, mag_ratio=1, add_margin=0.1,
    #                                       text_threshold=0.5, low_text=0.4, width_ths=0.9, height_ths=0.5,
    #                                       ycenter_ths=0.5, slope_ths=0.1)

    # cv.rectangle(cv_image, (0, 120), (100, 310), (0, 0, 255), 2)

    # crop_img = cv_image[120:310, 0:100]
    # os.makedirs("images\\cropped", exist_ok=True)
    # cv.imwrite(f"images\\cropped\\test.png", crop_img)

    # for i, box in enumerate(horizontal_list):
    #     x_min, x_max, y_min, y_max = box
    #     cv.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    #
    #     crop_img = cv_image[y_min:y_max, x_min:x_max]
    #     os.makedirs("images\\cropped", exist_ok=True)
    # cv.imwrite(f"images\\cropped\\{i}.png", crop_img)
    # break

    # cv.imshow('image', cv_image)

    # return ""


if __name__ == "__main__":
    infer(r"C:\Users\megis\PycharmProjects\OCR_Research\images\cropped\test\Any1.jpeg")
    start = time()
    infer(r"C:\Users\megis\PycharmProjects\OCR_Research\images\cropped\test\Any1.jpeg")
    print(f"Time: {time() - start}")

#
# # 128,165    576, 253
# crop_img = cv_image[165:253, 128:576]
# # os.makedirs("images\\cropped", exist_ok=True)
# cv2.imwrite(f"images\\cropped\\test.png", crop_img)
#
# crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#
#
# reader = easyocr.Reader(["en"], gpu=False)
# start = time()
# results = reader.readtext(crop_img, detail=1, batch_size=8)
# print(f"Time: {time() - start}")
# draw = False
# for lst, string, confidence in results:
#     print(lst)
#     print(string)
#     print(confidence)
#     if draw:
#         x1, y1 = lst[0][:2]  # Top-left corner
#         x2, y2 = lst[2][:2]  # Bottom-right corner
#
#         # Ensure the coordinates are integers
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#         # Crop the region of interest from the original image
#         crop_img_tmp = crop_img[y1:y2, x1:x2]
#         # cv2.rectangle(crop_img, tuple(lst[0]), tuple(lst[2]), (0, 255, 0), 2)
#         # cv2.imwrite(f"images\\cropped\\test.png", crop_img_tmp)
#         cv2.imshow('image', crop_img_tmp)
#         cv2.waitKey(0)
#
