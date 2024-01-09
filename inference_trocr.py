# # https://github.com/JaidedAI/EasyOCR
# # https://arxiv.org/pdf/1507.05717.pdf
# # https://arxiv.org/pdf/1904.01941.pdf
#
#
# import re
# import easyocr
from PIL import Image
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2GRAY, IMWRITE_JPEG_QUALITY
#
#
# reader=easyocr.Reader(["en"], gpu=True)
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
#     cv_image = cvtColor(cv_image, COLOR_BGR2GRAY)
#
#     # if not image_path.endswith(target_format):
#     #     imwrite('image.jpg', cv_image, [int(IMWRITE_JPEG_QUALITY), 100])
#
#     # if scale != 1:
#     #     cv_image = cv_image.resize(cv_image.shape[0] * scale, cv_image.shape[1] * scale)
#
#     return re.sub(r"\s+", " ", " ".join(reader.readtext(cv_image, detail=detail, batch_size=8))).strip().lower()
#
#
# if __name__ == "__main__":
#     print(infer(r"images\1695573507847.png", scale=1))


from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from numpy import ndarray, expand_dims, newaxis

model_version = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(model_version)
model = VisionEncoderDecoderModel.from_pretrained(model_version)


def infer(image_path: str, scale=1, target_format="png") -> str:
    """
    Infer text from image using TrOCR.
    :param image_path: Path to an image
    :param scale: Scale factor for image
    :param target_format: Target format for image
    :return: text
    """

    # TODO: First. Extract text boxes from image using OpenCV
    # TODO: Second. Crop text boxes from image
    # TODO: Third. Infer text from text boxes, bottom code for each text box and merge them
    image = Image.open(image_path).convert("RGB")
    # image = image.resize((int(image.width * scale), int(image.height * scale)))
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


if __name__ == "__main__":
    print(infer(r"images\1695573507847.png", scale=1))
