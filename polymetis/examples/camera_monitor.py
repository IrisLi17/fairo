from polymetis import CameraInterface
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    camera = CameraInterface(ip_address="101.6.103.171")
    (image, timestamp) = camera.read_once()
    n_channel = image.shape[-1]
    if n_channel == 3:
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = plt.subplots(1, 2)
    while True:
        (image, timestamp) = camera.read_once()
        if n_channel == 3:
            ax.cla()
            ax.imshow(image.astype(np.uint8))
            plt.pause(0.1)
            print(timestamp)
        else:
            ax[0].cla()
            ax[0].imshow(image[..., :3].astype(np.uint8))
            ax[1].cla()
            ax[1].imshow(image[..., -1])
            plt.pause(0.1)
            print(timestamp)