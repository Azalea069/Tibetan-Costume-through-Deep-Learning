# import os
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Used to crop images into squares

def crop_center(image):
    # Original image shape
    shape = image.shape
    # New shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1]-shape[2], 0) // 2
    offset_x = max(shape[2]-shape[1], 0) // 2
    # Return new image
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image

# Load and preprocess images

def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    # Cache image file
    # image_path = tf.keras.utils.get_file(
    #    os.path.basename(image_url)[-128:], image_url)
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0,1].
    img = tf.io.decode_image(
        tf.io.read_file(image_url),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

# Display images

def show_n(images, titles=('',)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    plt.show()

content_image_url = 'C:/Users/20721/Desktop/Tibetans/charlie.jpg'
style_image_url = 'C:/Users/20721/Desktop/Tibetans/Wei Zang/37.jpg'
output_image_size = 384

# Adjust content image size
content_img_size = (output_image_size, output_image_size)
# Style image size
style_img_size = (256, 256)
# Load images
content_image = load_image(content_image_url, content_img_size)
style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(
    style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
# Display images
# show_n([content_image, style_image], ['Content image', 'Style image'])

# Load model
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
# Style transfer
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]
# Display stylized image
show_n([content_image, style_image, stylized_image], titles=[
        'Original content image', 'Style image', 'Stylized image'])
plt.show()  # Show images