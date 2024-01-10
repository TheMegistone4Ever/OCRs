# https://github.com/sirfz/tesserocr


from PIL import Image, ImageDraw
from tesserocr import PyTessBaseAPI, RIL


def infer(image_path: str, scale=1, target_format="png", language="eng") -> str:
    """
    Infer text from image using TesserOCR.
    :param image_path: Path to an image
    :param scale: Scale factor for image
    :param target_format: Target format for image
    :param language: Language for OCR
    :return: text
    """
    path_data = "tessdata"

    # Apply scaling if needed
    if scale != 1:
        # You may need to install the 'pillow' library for this operation
        img = Image.open(image_path)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size)
        img.save(image_path, format=target_format)

    with PyTessBaseAPI(path=path_data, lang=language) as api:
        api.SetImageFile(image_path)
        text = api.GetUTF8Text()

        # Uncomment the following section if you want to print bounding boxes and confidence levels
        # boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        # print('Found {} textline image components.'.format(len(boxes)))
        # for i, (im, box, _, _) in enumerate(boxes):
        #     api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
        #     text = api.GetUTF8Text()
        #     conf = api.MeanTextConf()
        #     print(f"Box[{i}]: x={box['x']}, y={box['y']}, w={box['w']}, h={box['h']}, conf: {conf}, text: {text}")
        #     draw = ImageDraw.Draw(im)
        #     draw.rectangle(((box['x'], box['y']), (box['x']+box['w'], box['y']+box['h'])), outline="red")
        #     im.show()

    return text.strip().lower()


if __name__ == "__main__":
    print(infer(r"images/1695573507847.png"))
