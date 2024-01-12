# https://github.com/sirfz/tesserocr


from PIL import Image, ImageDraw
from tesserocr import PyTessBaseAPI, RIL
from os import remove
from os.path import splitext


def infer(image_path: str, scale=1, language="eng") -> str:
    """
    Infer text from image using TesserOCR.
    :param image_path: Path to an image
    :param scale: Scale factor for image
    :param language: Language for OCR
    :return: text
    """
    path_data = "tessdata"

    img = Image.open(image_path)
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    img = img.resize(new_size)

    if img.mode == 'RGBA':
        img = img.convert('RGB')

    path = image_path + ".resized" + splitext(image_path)[1]
    img.save(path)
    with PyTessBaseAPI(path=path_data, lang=language) as api:
        api.SetImageFile(path)
        text = api.GetUTF8Text()
    remove(path)

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
