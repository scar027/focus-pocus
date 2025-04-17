import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.datasets import cifar10

def add_gaussian_noise(images, mean=0.0, std=0.1):
  noise = np.random.normal(mean, std, images.shape)
  noisy = np.clip(images + noise, 0.0, 1.0)
  return noisy

def add_salt_pepper_noise(image, amount=0.05):
  noisy = image.copy()
  num_salt = int(np.ceil(amount * image.size * 0.5))
  num_pepper = int(np.ceil(amount * image.size * 0.5))

  # Salt
  coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
  noisy[coords[0], coords[1], :] = 1

  # Pepper
  coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
  noisy[coords[0], coords[1], :] = 0

def main():
  (x_train, _), (x_test, _) = cifar10.load_data()

  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255
  print("Shape of training set: ", x_train.shape)
  print("Shape of test set: ", x_test.shape)

if __name__ == "__main__":
  main()