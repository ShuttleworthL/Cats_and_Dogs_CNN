import matplotlib.pyplot as plt #visual representation and cost + accuracy graphing
import numpy as np #no need for cupy, only reading the npz file

#load model
model = np.load("CNN_mnist_model.npz")

t = int(model["t"]) #batch count

cost_past =  model["cost_past"] #for graphing cost and loss over time against number of batches
accuracy_past =  model["accuracy_past"]

x = []
for i in range(len(cost_past)):
    batches = 50*i #50 because cost and accuracy are recorded every 50 batches
    x.append(batches)

plt.figure(figsize=(10, 8))

# Cost graph
plt.subplot(2, 1, 1) #2 rows and 1 column of plots (plot to row 1)

plt.title("Cost over batches")
plt.xlabel("Number of batches")
plt.ylabel("Cost")

#-note first value for cost is set to 1 to avoid the initial cost spike (therefore is not truely representitve of the very first cost value) (this allows the cost to stabilize for the first 50 batches)
plt.plot(x, cost_past, color="red", linewidth=2)


# Accuracy graph
plt.subplot(2, 1, 2) #(plot to row 2)

plt.title("Accuracy over batches")
plt.xlabel("Number of batches")
plt.ylabel("Accuracy")

plt.plot(x, accuracy_past, color="blue", linewidth=2)

plt.tight_layout() #correctly layout subplots

plt.savefig("Cost+Accuracy_graph.png", dpi=300, bbox_inches="tight") #save graph as png (dpi-dots per inch/effective resolution)

plt.show()