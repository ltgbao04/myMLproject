import matplotlib.pyplot as plt

# create a figure and some subplots
fig, ax = plt.subplots()

# plot some data
ax.plot([1, 2, 3, 4])

# display the figure
plt.show()

import numpy as np

def visualize(X_train, Y_train, model):
    plt.scatter(X_train, Y_train)
    plt.xlabel("Age")
    plt.ylabel("Annual Income (k$)")
    x = np.array([min(X_train), max(X_train)])
    y = model.predict(x)
    plt.plot(x, y, color = "r")
    plt.show()