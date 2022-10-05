import pickle
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1)
with open("demo_first.pkl", "rb") as f:
    while True:
        try:
            data = pickle.load(f)
            image = data["image"]
            ax.cla()
            ax.imshow(image)
            plt.pause(0.1)
        except EOFError:
            break
