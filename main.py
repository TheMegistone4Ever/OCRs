import time
import subprocess
import matplotlib.pyplot as plt
import psutil
from inference_easyocr import infer as infer_easyocr
from inference_pytesseract import infer as infer_pytesseract
from inference_fast import infer as infer_fast
from inference_doctr import infer as infer_doctr
from inference_tesserocr import infer as infer_tesserocr
import os
import textwrap
from glob import glob


def directory_gen(path):
    while True:
        for entry in os.listdir(path):
            yield os.path.join(path, entry)


def measure_execution_time(infer_function, image_path, iters, directory, **kwargs):
    print(f"Warmup: {infer_function.__module__}, method: {infer_function.__name__}...")
    if directory:
        file = next(directory_gen(image_path))
        for iteration in range(5):
            infer_function(file, **kwargs)
    else:
        for iteration in range(5):
            infer_function(image_path, **kwargs)

    times = []
    cpu_usage = []
    ram_usage = []
    gpu_usage = []

    if directory:
        image_generator = directory_gen(image_path)
        for iteration in range(iters):
            next_img = next(image_generator)
            start_time = time.time()
            infer_function(next_img, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)

            # Profiling CPU and RAM usage
            process = psutil.Process(os.getpid())
            cpu_usage.append(process.cpu_percent())
            ram_usage.append(process.memory_info().rss / 1024 / 1024)  # in MB

            # Profiling GPU usage
            gpu_info = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
            gpu_usage.append(float(gpu_info.strip()))

            print(f"Module: {infer_function.__module__}, method: {infer_function.__name__}. "
                  f"Iteration {iteration + 1} of {iters} completed...")
    else:
        times = []
        cpu_usage = []
        ram_usage = []
        gpu_usage = []
        for iteration in range(iters):
            start_time = time.time()
            infer_function(image_path, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)

            # Profiling CPU and RAM usage
            process = psutil.Process(os.getpid())
            cpu_usage.append(process.cpu_percent())
            ram_usage.append(process.memory_info().rss / 1024 / 1024)  # in MB

            # Profiling GPU usage
            gpu_info = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
            gpu_usage.append(float(gpu_info.strip()))

            print(f"Module: {infer_function.__module__}, method: {infer_function.__name__}. "
                  f"Iteration {iteration + 1} of {iters} completed...")

    average_time = sum(times) / iters
    average_cpu = sum(cpu_usage) / iters
    average_ram = sum(ram_usage) / iters
    average_gpu = sum(gpu_usage) / iters

    print(f"Module: {infer_function.__module__}, method: {infer_function.__name__}. "
          f"Average execution time: {average_time:,.4f} seconds.")
    print(f"Average CPU usage: {average_cpu:.2f}%")
    print(f"Average RAM usage: {average_ram:.2f} MB")
    print(f"Average GPU usage: {average_gpu:.2f}%")

    return average_time, average_cpu, average_ram, average_gpu


def main(path_png, path_jpeg, directory=False, iterations=2):
    # Measure execution times for each inference function with PNG format
    easyocr_png_time, easyocr_png_cpu, easyocr_png_ram, easyocr_png_gpu = measure_execution_time(infer_easyocr, path_png, iterations, directory)
    fast_png_time, fast_png_cpu, fast_png_ram, fast_png_gpu = measure_execution_time(infer_fast, path_png, iterations, directory)
    pytesseract_png_time, pytesseract_png_cpu, pytesseract_png_ram, pytesseract_png_gpu = measure_execution_time(infer_pytesseract, path_png, iterations, directory, scale=3)
    doctr_png_time, doctr_png_cpu, doctr_png_ram, doctr_png_gpu = measure_execution_time(infer_doctr, path_png, iterations, directory)
    tesserocr_png_time, tesserocr_png_cpu, tesserocr_png_ram, tesserocr_png_gpu = measure_execution_time(infer_tesserocr, path_png, iterations, directory, scale=3)

    # Measure execution times for each inference function with JPG format
    easyocr_jpeg_time, easyocr_jpeg_cpu, easyocr_jpeg_ram, easyocr_jpeg_gpu = measure_execution_time(infer_easyocr, path_jpeg, iterations, directory)
    fast_jpeg_time, fast_jpeg_cpu, fast_jpeg_ram, fast_jpeg_gpu = measure_execution_time(infer_fast, path_jpeg, iterations, directory)
    pytesseract_jpeg_time, pytesseract_jpeg_cpu, pytesseract_jpeg_ram, pytesseract_jpeg_gpu = measure_execution_time(infer_pytesseract, path_jpeg, iterations, directory, scale=3)
    doctr_jpeg_time, doctr_jpeg_cpu, doctr_jpeg_ram, doctr_jpeg_gpu = measure_execution_time(infer_doctr, path_jpeg, iterations, directory)
    tesserocr_jpeg_time, tesserocr_jpeg_cpu, tesserocr_jpeg_ram, tesserocr_jpeg_gpu = measure_execution_time(infer_tesserocr, path_jpeg, iterations, directory, scale=3)

    # Plotting the results
    labels = ["EasyOCR", "FAST", "PyTesseract", "Doctr", "TesserOCR"]
    png_times = [easyocr_png_time, fast_png_time, pytesseract_png_time, doctr_png_time, tesserocr_png_time]
    jpeg_times = [easyocr_jpeg_time, fast_jpeg_time, pytesseract_jpeg_time, doctr_jpeg_time, tesserocr_jpeg_time]

    bar_width = 0.2
    index = range(len(labels))

    plot_bar, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(index, png_times, bar_width, label="PNG")
    ax.bar([bar + bar_width for bar in index], jpeg_times, bar_width, label="JPEG")

    ax.set_xticks([bar + bar_width / 2 for bar in index])
    ax.set_xticklabels(labels)
    ax.set_xlabel("Inference Functions")

    ax.set_ylabel(f"Average Time (s) for {iterations} Iterations")
    ax.set_title("Average Execution Time for Each Inference Function and Image Format")
    ax.legend()

    for i, value in enumerate(png_times):
        ax.text(i - 0.05, value + 0.01, f'{value:,.4f}', ha='center', va='bottom', fontsize=8)

    for i, value in enumerate(jpeg_times):
        ax.text(i + 0.05 + bar_width, value + 0.01, f'{value:,.4f}', ha='center', va='bottom', fontsize=8)

    plt.show()
    plt.close()


if __name__ == "__main__":
    # main(r"images\1695573507847.png", r"images\1695573507847.jpeg")
    # main(r"images\cropped\s.png", r"images\cropped\s.jpeg")

    main(r"images\cropped\test", r"images\cropped\test", directory=True)
    ocr_function = infer_easyocr
    images = glob(r"images\cropped\test\*.png")
    picture_grid, axs = plt.subplots(len(images) // 3, 3, figsize=(10, 10))
    picture_grid.suptitle(f"{ocr_function.__module__}'s OCR outcomes:", fontsize=26)
    for i, img_path in enumerate(images):
        img = plt.imread(img_path)
        ocr_result = ocr_function(img_path)
        axs[i // 3, i % 3].imshow(img)
        axs[i // 3, i % 3].axis("off")
        wrapped_text = textwrap.fill(ocr_result, width=20)
        axs[i // 3, i % 3].set_title(wrapped_text, fontsize=20)
    plt.show()
