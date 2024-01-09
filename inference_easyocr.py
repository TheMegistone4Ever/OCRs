# https://github.com/JaidedAI/EasyOCR
# https://arxiv.org/pdf/1507.05717.pdf
# https://arxiv.org/pdf/1904.01941.pdf


import re
import easyocr
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2GRAY, IMWRITE_JPEG_QUALITY


reader=easyocr.Reader(["en"], gpu=True)
def infer(image_path: str, scale=1, detail=0, target_format="png") -> str:
    """
    Infer text from image using EasyOCR.
    :param image_path: Path to an image
    :param scale: Scale factor for image
    :param detail: 0 - low, 1 - medium, 2 - high
    :param target_format: Target format for image
    :return: text
    """
    cv_image = imread(image_path)
    cv_image = cvtColor(cv_image, COLOR_BGR2GRAY)

    # if not image_path.endswith(target_format):
    #     imwrite('image.jpg', cv_image, [int(IMWRITE_JPEG_QUALITY), 100])

    # if scale != 1:
    #     cv_image = cv_image.resize(cv_image.shape[0] * scale, cv_image.shape[1] * scale)

    return re.sub(r"\s+", " ", " ".join(reader.readtext(cv_image, detail=detail, batch_size=8))).strip().lower()


if __name__ == "__main__":
    print(infer(r"images\1695573507847.png", scale=1))
