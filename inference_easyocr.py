# https://github.com/JaidedAI/EasyOCR
# https://arxiv.org/pdf/1507.05717.pdf
# https://arxiv.org/pdf/1904.01941.pdf


import re
import easyocr
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2GRAY, IMWRITE_JPEG_QUALITY
import cv2 as cv
import os


#reader=easyocr.Reader(["en"], gpu=True)
# def infer(image_path: str, scale=1, detail=0, target_format="png") -> str:
#     """
#     Infer text from image using EasyOCR.
#     :param image_path: Path to an image
#     :param scale: Scale factor for image
#     :param detail: 0 - low, 1 - medium, 2 - high
#     :param target_format: Target format for image
#     :return: text
#     """
#     cv_image = imread(image_path)
#     #cv_image = cvtColor(cv_image, COLOR_BGR2GRAY)
#
#     # if not image_path.endswith(target_format):
#     #     imwrite('image.jpg', cv_image, [int(IMWRITE_JPEG_QUALITY), 100])
#
#     # if scale != 1:
#     #     cv_image = cv_image.resize(cv_image.shape[0] * scale, cv_image.shape[1] * scale)
#
#     # return re.sub(r"\s+", " ", " ".join(reader.readtext(cv_image, detail=detail, batch_size=8))).strip().lower()
#
#     #[[horizontal_list], _] = reader.detect(cv_image, min_size=1, canvas_size=2560, mag_ratio=1, add_margin=0.1,
#     #                                       text_threshold=0.5, low_text=0.4, width_ths=0.9, height_ths=0.5,
#     #                                       ycenter_ths=0.5, slope_ths=0.1)
#
#     cv.rectangle(cv_image, (0, 120), (100, 310), (0, 0, 255), 2)
#
#     crop_img = cv_image[120:310, 0:100]
#     # os.makedirs("images\\cropped", exist_ok=True)
#     cv.imwrite(f"images\\cropped\\test.png", crop_img)
#
#
#
#     for i, box in enumerate(horizontal_list):
#         x_min, x_max, y_min, y_max = box
#         cv.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
#
#         crop_img = cv_image[y_min:y_max, x_min:x_max]
#         # os.makedirs("images\\cropped", exist_ok=True)
#         cv.imwrite(f"images\\cropped\\{i}.png", crop_img)
#         break
#
#     # cv.imshow('image', cv_image)
#
#     return ""


#if __name__ == "__main__":

cv_image = cv.imread("images/1695573507847.png")

# 128,165    576, 253
crop_img = cv_image[165:253, 128:576]
# os.makedirs("images\\cropped", exist_ok=True)
cv.imwrite(f"images\\cropped\\test.png", crop_img)

crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)

reader = easyocr.Reader(["en"], gpu=True)

results = reader.readtext(crop_img, detail=1, batch_size=8)
draw = True
for lst, string, confidence in results:
    if draw:
        cv.rectangle(crop_img, tuple(lst[0]), tuple(lst[2]), (0, 255, 0), 2)
        cv.imwrite(f"images\\cropped\\test.png", crop_img)
        # cv.imshow('image', crop_img)
    print(lst)
    print(string)
    print(confidence)
