import time
import matplotlib.pyplot as plt
from inference_easyocr import infer as infer_easyocr
from inference_fast import infer as infer_fast
from inference_pytesseract import infer as infer_pytesseract
from inference_doctr import infer as infer_doctr


def measure_execution_time(infer_function, image_path, iters=10, **kwargs):
    times = []
    for _ in range(iters):
        start_time = time.time()
        infer_function(image_path, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    average_time = sum(times) / iters
    return average_time


image_path_png, image_path_jpg = r"images\1695573507847.png", r"images\1695573507847.jpg"
iterations = 50

# Measure execution times for each inference function with PNG format
easyocr_png_time = measure_execution_time(infer_easyocr, image_path_png, iterations, target_format="png")
fast_png_time = measure_execution_time(infer_fast, image_path_png, iterations, target_format="png")
pytesseract_png_time = measure_execution_time(infer_pytesseract, image_path_png, iterations, target_format="png",
                                              scale=3)
doctr_png_time = measure_execution_time(infer_doctr, [image_path_png], iterations, target_format="png")

# Measure execution times for each inference function with JPG format
easyocr_jpg_time = measure_execution_time(infer_easyocr, image_path_jpg, iterations, target_format="jpg")
fast_jpg_time = measure_execution_time(infer_fast, image_path_jpg, iterations, target_format="jpg")
pytesseract_jpg_time = measure_execution_time(infer_pytesseract, image_path_jpg, iterations, target_format="jpg",
                                              scale=3)
doctr_jpg_time = measure_execution_time(infer_doctr, [image_path_jpg], iterations, target_format="jpg")

# Plotting the results
labels = ["EasyOCR", "FAST", "PyTesseract", "Doctr"]
png_times = [easyocr_png_time, fast_png_time, pytesseract_png_time, doctr_png_time]
jpg_times = [easyocr_jpg_time, fast_jpg_time, pytesseract_jpg_time, doctr_jpg_time]

bar_width = 0.2
index = range(len(labels))

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
bar1 = ax.bar(index, png_times, bar_width, label="PNG")
bar2 = ax.bar([i + bar_width for i in index], jpg_times, bar_width, label="JPG")

ax.set_xticks([i + bar_width/2 for i in index])
ax.set_xticklabels(labels)
ax.set_xlabel("Inference Functions")

ax.set_ylabel(f"Average Time (s) for {iterations} Iterations")
ax.set_title("Average Execution Time for Each Inference Function and Image Format")
ax.legend()

plt.show()
