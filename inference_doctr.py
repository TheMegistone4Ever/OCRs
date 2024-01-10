from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def infer(image_paths: list, show: bool = False, scale: int = 1, target_format: str = "png",
          model=ocr_predictor(pretrained=True)) -> str:
    """
    Infer text from a list of images using doctr library.
    :param image_paths: List of image paths
    :param show: Whether to display the result
    :param scale: Scale factor for image
    :param target_format: Target format for image
    :param model: Doctr model
    :return: Text
    """
    pages = DocumentFile.from_images(image_paths)
    result = model(pages)

    words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    words.append(word.render())

    if show:
        result.show(pages)

    return " ".join(words)


if __name__ == "__main__":
    images = [r"images\1695573507847.png"]
    print(infer(images, show=True))
