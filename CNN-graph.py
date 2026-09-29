import matplotlib.pyplot as plt #visual representation and cost + accuracy graphing
import numpy as np #no need for cupy, only reading the npz file + averages

#load model
model = np.load("CNN_mnist_model.npz")

t = int(model["t"]) #batch count

cost_past =  model["cost_past"] #for graphing cost and loss over time against number of batches
accuracy_past =  model["accuracy_past"]
validation_average_past = model["validation_average_past"]

cost_past_smooth = []
accuracy_past_smooth = []
x_smooth = []

x = []
for i in range(len(cost_past)):
    batch = i
    x.append(batch)
    if cost_past[i] > 1 and i <= 50:
        cost_past[i] = 1 #capp cost out at 1 in first 50 batches

smooth_window = 50 #how many values are included in the average (more=smoother)

cost_past_smooth = np.convolve( #running average/convoloution
    cost_past, #data to be convolved
    np.ones(smooth_window) / smooth_window, #filter to convolve (divide eahc number by 50 then add all 50 numbers together-running average)
    mode="valid" #uses valid convolution (no padding)
)

accuracy_past_smooth = np.convolve( #same method as above but on past accuracy data
    accuracy_past,
    np.ones(smooth_window) / smooth_window,
    mode="valid"
)

x_smooth = np.arange(smooth_window - 1, len(cost_past)) #arrange values starting from the first smooth window to the end of the data (the start is cut of without enough to average with)

plt.figure(figsize=(10, 8))

# Cost graph
plt.subplot(2, 1, 1) #2 rows and 1 column of plots (plot to row 1)

plt.title("Cost over batches")
plt.xlabel("Number of batches")
plt.ylabel("Cost")

#-note first value for cost is set to 1 to avoid the initial cost spike (therefore is not truely representitve of the very first cost value) (this allows the cost to stabilize for the first 50 batches)
plt.plot(x, cost_past, color="red", linewidth=2)
#plot smoothed line after(over the top of) per batch data
plt.plot(x_smooth, cost_past_smooth, color='green') #(green - for max contrast)


# Accuracy graph
plt.subplot(2, 1, 2) #(plot to row 2)

plt.title("Accuracy over batches")
plt.xlabel("Number of batches")
plt.ylabel("Accuracy")

plt.plot(x, accuracy_past, color="blue", linewidth=2)
#plot smoothed line after(over the top of) per batch data
plt.plot(x_smooth, accuracy_past_smooth, color='orange') #(orange - for max contrast)

for i in range(len(validation_average_past)):
    plt.plot((i+1)*22500, validation_average_past[i], 'ks', markersize=8) #(i+1)*22500 to correctly plot the dot(black square) at the emd of the batch that triggered the validation

plt.tight_layout() #correctly layout subplots

plt.savefig("Cost+Accuracy_graph.png", dpi=300, bbox_inches="tight") #save graph as png (dpi-dots per inch/effective resolution)

plt.show()