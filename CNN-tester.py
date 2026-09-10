import cupy as np #for vector and matrix math (on gpu)
import numpy
from cupy.lib.stride_tricks import sliding_window_view #more efficient way for locating regions within a image (complied c code) + (cupy gpu version)

#number of training examples before computing cost and backprop

dataset = []

for i in range(12500):
    dataset.append((f"Cat_{i}", 1))  # 1 = cat
    dataset.append((f"Dog_{i}", 0))  # 0 = dog

#<==========functions==========>

#probability distibution 0-1
def softmax(x): #probability distibution 0-1
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def sigmoid(x): #condenses values into 0-1 activation
    return 1/ (1 + np.exp(-x)) #1/1 + e**-x

def ReLU(x):
    return np.maximum(0, x)

def accurate(x, lable):
    highest = int(np.argmax(x))
    return lable == highest

def maxpool2x2(feature_maps):
    windows = sliding_window_view(feature_maps, (2,2), axis=(1,2)) #finding every 2x2 position in feature map giving C,W,H,2,2 where c is feature maps and w and h is width and height
    windows = windows[:, ::2, ::2] #dividing positions by 2 (removing every other 2x2 region to prevent overlap - C,W/2,h/2,2,2)
    pooled = windows.max(axis=(3,4)) #finding max value from every 2x2 region and discarding rest - C,W/2,H/2 where each position is the max value

    winner = windows.reshape(*windows.shape[:3], 4).argmax(axis=3) #condense last 2x2 into 4, then agrmax gives the position in the 4 region of the max value (keep first 3 shapes the same)
    mask = np.eye(4)[winner] #expands the winning index back into 4 locations where the max value is 1 - e.g. max value index = 2, 0,0,1,0
    #eye function creates arrays each with length 4, with each array having the 1 value in a diffrent position, we then select witch array we want (witch position we want the 1 in)
    #shape - C,W/2,H/2,4 where 4 is just the arasy of 0s with a 1 in one position
    mask = mask.reshape(
        feature_maps.shape[0],
        pooled.shape[1],
        pooled.shape[2],
        2,
        2
    ) #returning array of length 4 containing one 1 and 3 0s into a 2x2 array, shape - C,W/2,H/2,2,2
    mask = mask.transpose(0,1,3,2,4) #shape - C,W/2,2,H/2,2
    pool_mask = mask.reshape(feature_maps.shape) #then reshape to C,W,H

    return pooled, pool_mask #output pooled values, and witch values where their maximum

#<==========init values==========>

#load model
model = numpy.load("CNN_mnist_model.npz")

#float 32 to decrease computation via data size whilst having negligible effect on accuracy

#retrieve model weights and bias + convert to gpu arrays
kernal = np.asarray(model["kernal"], dtype=np.float32)
kernal_bias = np.asarray(model["kernal_bias"], dtype=np.float32)
kernal_2 = np.asarray(model["kernal_2"], dtype=np.float32)
kernal_bias_2 = np.asarray(model["kernal_bias_2"], dtype=np.float32)
kernal_3 = np.asarray(model["kernal_3"], dtype=np.float32)
kernal_bias_3 = np.asarray(model["kernal_bias_3"], dtype=np.float32)
kernal_4 = np.asarray(model["kernal_4"], dtype=np.float32)
kernal_bias_4 = np.asarray(model["kernal_bias_4"], dtype=np.float32)

weights_input_hidden = np.asarray(model["weights_input_hidden"], dtype=np.float32)
bias_hidden = np.asarray(model["bias_hidden"], dtype=np.float32)
weights_hidden_hidden2 = np.asarray(model["weights_hidden_hidden2"], dtype=np.float32)
bias_hidden2 = np.asarray(model["bias_hidden2"], dtype=np.float32)
weights_hidden2_output = np.asarray(model["weights_hidden2_output"], dtype=np.float32)
bias_output = np.asarray(model["bias_output"], dtype=np.float32)
    
dataset_accuracy = []

for i in range(1000):

    #<==========conv layer 1==========>

    file_name, training_labels_data = dataset[i]
    training_images_data = np.asarray( #convert to gpu
        numpy.load(f"Processed/{file_name}.npy")
    )
    #add 1 layer of padding to each side of the image matrix (allows a 3x3 kernal to passover every pixel)
    padded_training_images_data = np.pad((training_images_data/255)-0.5, 1, mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)
    #divided by 255 to normilize in 0-1, then -0.5 to center around 0

    output = np.zeros((10, 240, 240)) #10 outputs for filtered images (each the same shape of original image)
    raw_conv1 = np.zeros((10, 240, 240))
    region = sliding_window_view(padded_training_images_data, (3,3))
    #e.g if x=0, y=0 output matrix of 3x3 region with the top left corner centered at 0,0
    region = region[None, :, :, :, :] #adjusting shapes to broadcast together
    kernals = kernal[:, None, None, :, :]

    #multiply each corosponding element in the matrices(kernal and region) then sum output matrix, then add kernal bias
    value = np.sum(region * kernals, axis=(3,4)) #sum accross axis 3 and 4 (3x3)
    raw_conv1 = value + kernal_bias[:, None, None] #adjusting shapes to broadcast correctly and add kernal bias accross all feature maps
    output = ReLU(raw_conv1) #value for filtered pixel (convoloution of original image and kernal) + activation function

    feature_maps, pool_mask = maxpool2x2(output) #max pooling 2x2 function (vectorised)

    #<==========conv layer 2==========>

    #add 1 layer of padding to each side of the image matrix (not the vector of image matricies) (allows a 3x3 kernal to passover every pixel)
    padded_previous_feature_maps = np.pad(feature_maps, ((0,0), (1,1), (1,1)), mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)

    output_2 = np.zeros((20, 120, 120)) #20 outputs for filtered images (each the same shape of original image)
    raw_conv2 = np.zeros((20, 120, 120))
    region = sliding_window_view(padded_previous_feature_maps, (3,3), axis=(1,2))
    #e.g if x=0, y=0 output matrix of 3x3 region with the top left corner centered at 0,0
    region = region[None, :, :, :, :, :] #adjusting shapes to broadcast correctly
    kernals = kernal_2[:, :, None, None, :, :]

    #multiply each corosponding element in the matrices(kernal and region) then sum output matrix, then add kernal bias
    raw_conv2 = np.sum(region * kernals, axis=(1,4,5)) + kernal_bias_2[:, None, None] #sum accross feature maps amd kernal values then add kernal bias accross all feature maps
    output_2 = ReLU(raw_conv2) #sum of values for filtered pixels across (convoloution of original image and kernal)

    feature_maps_2, pool2_mask = maxpool2x2(output_2) #max pooling 2x2 function (vectorised)

    #<==========conv layer 3==========>

    #add 1 layer of padding to each side of the image matrix (not the vector of image matricies) (allows a 3x3 kernal to passover every pixel)
    padded_previous_feature_maps = np.pad(feature_maps_2, ((0,0), (1,1), (1,1)), mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)

    output_3 = np.zeros((40, 60, 60)) #40 outputs for filtered images (each the same shape of original image)
    raw_conv3 = np.zeros((40, 60, 60))
    region = sliding_window_view(padded_previous_feature_maps, (3,3), axis=(1,2))
    #e.g if x=0, y=0 output matrix of 3x3 region with the top left corner centered at 0,0
    region = region[None, :, :, :, :, :] #adjusting shapes to broadcast correctly
    kernals = kernal_3[:, :, None, None, :, :]

    #multiply each corosponding element in the matrices(kernal and region) then sum output matrix, then add kernal bias
    raw_conv3 = np.sum(region * kernals, axis=(1,4,5)) + kernal_bias_3[:, None, None] #sum accross feature maps amd kernal values then add kernal bias accross all feature maps
    output_3 = ReLU(raw_conv3) #sum of values for filtered pixels across (convoloution of original image and kernal)

    feature_maps_3, pool3_mask = maxpool2x2(output_3) #max pooling 2x2 function (vectorised)

    #<==========conv layer 4==========>

    #add 1 layer of padding to each side of the image matrix (not the vector of image matricies) (allows a 3x3 kernal to passover every pixel)
    padded_previous_feature_maps = np.pad(feature_maps_3, ((0,0), (1,1), (1,1)), mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)

    output_4 = np.zeros((80, 30, 30)) #80 outputs for filtered images (each the same shape of original image)
    raw_conv4 = np.zeros((80, 30, 30))
    region = sliding_window_view(padded_previous_feature_maps, (3,3), axis=(1,2))
    #e.g if x=0, y=0 output matrix of 3x3 region with the top left corner centered at 0,0
    region = region[None, :, :, :, :, :] #adjusting shapes to broadcast correctly
    kernals = kernal_4[:, :, None, None, :, :]

    #multiply each corosponding element in the matrices(kernal and region) then sum output matrix, then add kernal bias
    raw_conv4 = np.sum(region * kernals, axis=(1,4,5)) + kernal_bias_4[:, None, None] #sum accross feature maps amd kernal values then add kernal bias accross all feature maps
    output_4 = ReLU(raw_conv4) #sum of values for filtered pixels across (convoloution of original image and kernal)

    feature_maps_4, pool4_mask = maxpool2x2(output_4) #max pooling 2x2 function (vectorised)

    input_layer = feature_maps_4.flatten()

    #<==========dense layers==========>

    #input data (18000 neurons)
    #flatten converts 80x15x15 matrix of grayscale values down to 1d 18000 long vector of grayscale values
    #scaled down grayscale values from 255 to 0-1 range (activation range)
    inputs = input_layer

    #activations

    #hidden layer values without ReLU
    hidden_raw = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_activation = ReLU(hidden_raw) #hidden layer activations

    #second hidden layer values without ReLU
    hidden2_raw = np.dot(hidden_activation, weights_hidden_hidden2) + bias_hidden2
    hidden2_activation = ReLU(hidden2_raw) #second hidden layer activations

    #output layer values without ReLU
    output_raw = np.dot(hidden2_activation, weights_hidden2_output) + bias_output
    output_activation = softmax(output_raw) #output layer activations

    accuracy = accurate(output_activation, training_labels_data)
    print(i,": ", accuracy)

    dataset_accuracy.append(accuracy)

dataset_accuracy = numpy.mean(dataset_accuracy)
print("total dataset accuracy: ", dataset_accuracy*100, "%")
