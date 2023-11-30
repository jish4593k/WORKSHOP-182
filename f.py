import cv2
import time
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

# Function to capture background using KMeans clustering
def capture_background_kmeans(cap):
    bg = 0
    for _ in range(60):
        ret, frame = cap.read()
        bg += frame
    bg /= 60
    return np.flip(bg.astype(np.uint8), axis=1)

# Function to detect dominant color in the image using KMeans clustering
def detect_dominant_color(image):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1).fit(pixels)
    return kmeans.cluster_centers_[0]


def remove_background(img, bg, color_threshold=50):
    diff = cv2.absdiff(img, bg)
    diff = np.sum(diff, axis=-1)
    mask = (diff > color_threshold).astype(np.uint8) * 255
    return cv2.bitwise_and(img, img, mask=mask)


def display_images(images, titles):
    plt.figure(figsize=(12, 6))
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, 2, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
    plt.show()

def main():
  
    root = tk.Tk()
    root.withdraw()
    bg_path = filedialog.askopenfilename(title="Select Background Image")

    cap = cv2.VideoCapture(0)
    time.sleep(2)
    background = cv2.imread(bg_path)
    background = np.flip(background, axis=1)

    output_file = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        img = np.flip(img, axis=1)

        object_img = remove_background(img, background)

        display_images([img, object_img], ["Original Image", "Object with Removed Background"])

        output_file.write(object_img)

        cv2.imshow("Magic", object_img)
        cv2.waitKey(1)

    cap.release()
    output_file.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
