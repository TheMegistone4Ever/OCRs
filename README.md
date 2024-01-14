# OCRs

This project aims to compare and evaluate various Optical Character Recognition (OCR) libraries for text extraction
from images. The supported OCR libraries include [EasyOCR](https://github.com/JaidedAI/EasyOCR),
[PyTesseract](https://github.com/madmaze/pytesseract), [Doctr](https://github.com/mindee/doctr),
[TesserOCR](https://github.com/sirfz/tesserocr), [FAST](https://github.com/czczup/FAST),
and [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr).

## Table of Contents

1. [Installation](#1-installation)
2. [Usage](#2-usage)
3. [Inference Functions](#3-inference-functions)
   - 3.1 [EasyOCR](#31-easyocr)
   - 3.2 [PyTesseract](#32-pytesseract)
   - 3.3 [Doctr](#33-doctr)
   - 3.4 [TesserOCR](#34-tesserocr)
   - 3.5 [FAST](#35-fast)
   - 3.6 [TrOCR](#36-trocr)
4. [License](#4-license)

## 1. Installation

To run this project, you need to install the required dependencies. You can use the following command:

```
pip install -r requirements.txt
```

## 2. Usage

To run the project, execute the main function in the `main.py` file:

```
python main.py
```

This will measure the execution times for each OCR library on sample images and generate a bar chart to visualize
the results.

## 3. Inference Functions

### 3.1. EasyOCR

EasyOCR is an OCR library based on the EasyOCR model. It provides a straightforward API for text extraction from
images. [Code](inference_easyocr.py) | [Paper](https://arxiv.org/pdf/1507.05717.pdf)

### 3.2. PyTesseract

PyTesseract is a Python wrapper for Google's Tesseract-OCR Engine. It is widely used for text extraction from images.
[Code](inference_pytesseract.py) |
[Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33418.pdf)

### 3.3. Doctr

Doctr is an OCR library that utilizes deep learning models for document analysis. It is suitable for extracting text
from documents. [Code](inference_doctr.py) | [Paper](https://arxiv.org/pdf/1707.03718.pdf)

### 3.4. TesserOCR

TesserOCR is a Python wrapper for Tesseract-OCR. It provides a simple interface for text extraction from images.
[Code](inference_tesserocr.py)

### 3.5. FAST

FAST is an OCR library based on the FAST model. It focuses on efficient and accurate text extraction from images.
[Code](inference_fast.py) | [Paper](https://arxiv.org/pdf/2111.02394.pdf)

### 3.6. TrOCR

TrOCR is an OCR library based on the TrOCR model. It uses transformers for text recognition and can handle various
languages. [Code](inference_trocr.py)

## 4. License

The project is licensed under the [CC BY-NC 4.0 License](LICENSE.md).