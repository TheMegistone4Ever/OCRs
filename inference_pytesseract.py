# https://github.com/madmaze/pytesseract
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33418.pdf

import re
from pytesseract import image_to_string
from PIL import Image


def infer(image_path: str, scale=1, target_format="png") -> str:
    """
    Infer text from image using Tesseract OCR.
    :param image_path: Path to an image
    :param scale: Scale factor for image
    :param target_format: Target format for image
    :return: text
    """
    pil_image = Image.open(image_path)
    # if pil_image.format != target_format:
    #     pil_image = pil_image.convert(target_format)
    if scale != 1:
        pil_image = pil_image.resize((pil_image.width * scale, pil_image.height * scale))
    pil_image = pil_image.convert("L")
    return re.sub(r"\s+", " ", image_to_string(pil_image)).strip().lower()


if __name__ == "__main__":
    print(infer(r"images/1695573507847.png"))
