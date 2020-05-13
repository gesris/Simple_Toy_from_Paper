import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1234)   
import tensorflow_probability as tfp
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

## Load Data
model = tf.keras.models.load_model('./mymodel')
plot_label = pickle.load(open("plot_label.pickle", "rb"))

# Create meshgrid for 2D plot
length = 1000
sensibility = 0.005
b = np.linspace(-3, 3, length)
xx, yy = np.meshgrid(b, b, sparse=False)
x = np.vstack([xx.reshape((length * length)), yy.reshape((length * length))]).T
c = tf.squeeze(model(x)).numpy()
c = c.reshape((length, length))

# visualize decision boundary
boundary = tf.squeeze(model(x)).numpy()
boundary = boundary.reshape((length, length))
# iterate through c to highlight decision boundary
for y in range(0, length):
    for x in range(0, length):
        if c[y][x] > 0.5-sensibility and c[y][x] < 0.5 + sensibility:
            boundary[y][x] = 1
        else:
            boundary[y][x] = 0

# Plot NN function
print("\nPlotting NN function\n")
limit = [-3, 3]
plt.figure(figsize=(7, 6))
cbar = plt.contourf(xx, yy, c + boundary, levels=np.linspace(c.min(), c.max(), 21))
plt.colorbar(cbar)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Neural network function")
plt.tight_layout()
plt.xlim(limit[0], limit[1])
plt.ylim(limit[0], limit[1])
plt.savefig("./plots/NN_function_{}.png".format(plot_label), bbox_inches = "tight")

