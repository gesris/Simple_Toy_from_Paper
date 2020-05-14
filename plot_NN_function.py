import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1234)
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", size=16, family="serif")
from tqdm import tqdm

## Load Data
model = tf.keras.models.load_model('./mymodel')
plot_label = pickle.load(open("plot_label.pickle", "rb"))


# Create meshgrid for 2D plot
length = 1000
sensibility = 0.005
b = np.linspace(-3, 5, length)
xx, yy = np.meshgrid(b, b, sparse=False)
x = np.vstack([xx.reshape((length * length)), yy.reshape((length * length))]).T
c = tf.squeeze(model(x)).numpy()
c = c.reshape((length, length))


# visualize decision boundary
print("\nPlotting NN function\n")
boundary = tf.squeeze(model(x)).numpy()
boundary = boundary.reshape((length, length))
# iterate through c to highlight decision boundary
for y in tqdm(range(0, length)):
    for x in range(0, length):
        if c[y][x] > 0.5-sensibility and c[y][x] < 0.5 + sensibility:
            boundary[y][x] = 1
        else:
            boundary[y][x] = 0


# Plot NN function
limit = [-3, 3]
plt.figure(figsize=(7, 6))
plt.grid()
cbar = plt.contourf(xx, yy, c + boundary, levels=np.linspace(c.min(), c.max(), 21))
cbar = plt.colorbar(format="%.1f")
cbar.set_label("$f$")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(limit[0], limit[1])
plt.ylim(limit[0], limit[1])
#ticks = [-2, 0, 2, 4]
#plt.gca().set_xticks(ticks)
#plt.gca().set_yticks(ticks)
plt.savefig("./plots/NN_function_{}.png".format(plot_label), bbox_inches = "tight")
#plt.show()
