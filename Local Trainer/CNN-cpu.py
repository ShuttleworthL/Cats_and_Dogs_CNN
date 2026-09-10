import numpy as np #for vector and matrix math
from numpy.lib.stride_tricks import sliding_window_view #more efficient way for locating regions within a image (complied c code)
import keyboard #for dynamic saving
import time #for measuring code time
import os #for checking for existing model

#number of training examples before computing cost and backprop
batch_size = 64

dataset = []

for i in range(12500):
    dataset.append((f"Cat_{i}", 1))  # 1 = cat
    dataset.append((f"Dog_{i}", 0))  # 0 = dog

# shuffle dataset order
np.random.shuffle(dataset)

#<==========functions==========>

#update adam optimizer (takes in the model params, current gradients, m(gradients direction), v(gradient magnitude) and t(number of previous gradients))
def update_adam(params, grads, m, v, t, learning_rate=0.002, beta1=0.9, beta2=0.99, epsilon=1e-8): #epsilon to prevent dividing by 0

    for i in range(len(params)): #for every model parameter
        #update moving weighted averages
        m[i] = beta1 * m[i] + (1-beta1) * grads[i] #weighted average of previous weighted average and current gradient
        v[i] = beta2 * v[i] + (1-beta2) * (grads[i] ** 2) #uses diffrent beta to compensate for squaring

        #bias correction
        #when computing first gradients the moving average is smaller than it should be becuse the previous gradients no longer exist so are 0
        #so this boosts gradients at the start while resolving to normal as number of gradients increases
        corrected_m = m[i] / (1-beta1 **t)
        corrected_v = v[i] / (1-beta2 **t)

        #scales parameters individually and * LR
        params[i] -= learning_rate *  corrected_m / (np.sqrt(corrected_v) + epsilon) #epsilon to prevent dividing by 0

    return params, m, v

#probability distibution 0-1
def softmax(x): #probability distibution 0-1
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def sigmoid(x): #condenses values into 0-1 activation
    return 1/ (1 + np.exp(-x)) #1/1 + e**-x

def ReLU(x):
    return np.maximum(0, x)

def cross_entropy_cost(output, label): #simplified cross entropy loss
    y = np.zeros(2)
    y[label] = 1
    batch_expected_output.append(y) #expected output

    minimum_probability = 1 * 10**-12 #stops values approching -infinity
    return -np.log(output[label] + minimum_probability)

def accurate(x, lable):
    highest = int(np.argmax(x))
    return lable == highest
 
def sigmoid_derivative(z): #derivative of sigmoid with respect to weighted sum
    s = sigmoid(z)
    return s * (1-s)

def ReLU_derivative(x):
    return (x > 0).astype(float) #for every element is it greater or less than 0, represented as 1 or 0 - (true or false)

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

if not os.path.exists("CNN_mnist_model.npz"):
    #float 32 to decrease computation via data size whilst having negligible effect on accuracy

    #10 kernals (all 3x3)
    kernal = (np.random.randn(10, 3, 3) * 0.1).astype(np.float32) #random init values for 10 3x3 kernal (filter) (small for first init values, hence 0.1) -(array of weights)
    kernal_bias = np.zeros(10, dtype=np.float32) #init bias values for each kernal

    #20 groups of 10 kernals (each 3x3)
    kernal_2 = (np.random.randn(20, 10, 3, 3) * 0.2).astype(np.float32) #random init values for 20 groups of 10 3x3 kernal (filter) (small for first init values, hence 0.1) -(array of weights)
    kernal_bias_2 = np.zeros(20, dtype=np.float32) #init bias values for each kernal

    #40 groups of 20 kernals (each 3x3)
    kernal_3 = (np.random.randn(40, 20, 3, 3) * 0.1).astype(np.float32) #random init values for 40 groups of 20 3x3 kernal (filter) (small for first init values, hence 0.1) -(array of weights)
    kernal_bias_3 = np.zeros(40, dtype=np.float32) #init bias values for each kernal

    #80 groups of 40 kernals (each 3x3)
    kernal_4 = (np.random.randn(80, 40, 3, 3) * 0.1).astype(np.float32) #random init values for 80 groups of 40 3x3 kernal (filter) (small for first init values, hence 0.1) -(array of weights)
    kernal_bias_4 = np.zeros(80, dtype=np.float32) #init bias values for each kernal

    #random starting weights and bias (input layer neurons do not have bias)

    #128 rows, 18000 columns, for 18000 input neurons to 128 hidden neurons e.g. [0.1, -0.3, 0.7]x128 ect ...
    #weights from input to hidden layer
    weights_input_hidden = (np.random.randn(18000, 128) * 0.1).astype(np.float32) #random array of weights in gaussian distrubution (scaled down for start values)
    bias_hidden = np.zeros(128, dtype=np.float32) #bias for 128 hidden neurons (0 for start values)

    #64 rows, 128 columns, for 128 hidden neurons to 64 second hidden layer neurons
    weights_hidden_hidden2 = (np.random.randn(128, 64) * 0.1).astype(np.float32)
    bias_hidden2 = np.zeros(64, dtype=np.float32) #bias for 64 second hidden layer neurons

    #2 rows, 64 columns, for 64 hidden neurons to 2 output neurons
    weights_hidden2_output = (np.random.randn(64, 2) * 0.1).astype(np.float32)
    bias_output = np.zeros(2, dtype=np.float32) #bias for 2 output neurons

    m_kernal = np.zeros_like(kernal)
    m_kernal_bias = np.zeros_like(kernal_bias)
    m_kernal_2 = np.zeros_like(kernal_2)
    m_kernal_bias_2 = np.zeros_like(kernal_bias_2)
    m_kernal_3 = np.zeros_like(kernal_3)
    m_kernal_bias_3 = np.zeros_like(kernal_bias_3)
    m_kernal_4 = np.zeros_like(kernal_4)
    m_kernal_bias_4 = np.zeros_like(kernal_bias_4)
    m_weights_input_hidden = np.zeros_like(weights_input_hidden)
    m_bias_hidden = np.zeros_like(bias_hidden)
    m_weights_hidden_hidden2 = np.zeros_like(weights_hidden_hidden2)
    m_bias_hidden2 = np.zeros_like(bias_hidden2)
    m_weights_hidden2_output = np.zeros_like(weights_hidden2_output)
    m_bias_output = np.zeros_like(bias_output)

    v_kernal = np.zeros_like(kernal)
    v_kernal_bias = np.zeros_like(kernal_bias)
    v_kernal_2 = np.zeros_like(kernal_2)
    v_kernal_bias_2 = np.zeros_like(kernal_bias_2)
    v_kernal_3 = np.zeros_like(kernal_3)
    v_kernal_bias_3 = np.zeros_like(kernal_bias_3)
    v_kernal_4 = np.zeros_like(kernal_4)
    v_kernal_bias_4 = np.zeros_like(kernal_bias_4)
    v_weights_input_hidden = np.zeros_like(weights_input_hidden)
    v_bias_hidden = np.zeros_like(bias_hidden)
    v_weights_hidden_hidden2 = np.zeros_like(weights_hidden_hidden2)
    v_bias_hidden2 = np.zeros_like(bias_hidden2)
    v_weights_hidden2_output = np.zeros_like(weights_hidden2_output)
    v_bias_output = np.zeros_like(bias_output)

    t = 0
    np.savez(
        "CNN_mnist_model.npz",
        kernal=kernal,
        kernal_bias=kernal_bias,
        kernal_2=kernal_2,
        kernal_bias_2=kernal_bias_2,
        kernal_3=kernal_3,
        kernal_bias_3=kernal_bias_3,
        kernal_4=kernal_4,
        kernal_bias_4=kernal_bias_4,
        weights_input_hidden=weights_input_hidden,
        bias_hidden=bias_hidden,
        weights_hidden_hidden2=weights_hidden_hidden2,
        bias_hidden2=bias_hidden2,
        weights_hidden2_output=weights_hidden2_output,
        bias_output=bias_output,

        m_kernal=m_kernal,
        m_kernal_bias=m_kernal_bias,
        m_kernal_2=m_kernal_2,
        m_kernal_bias_2=m_kernal_bias_2,
        m_kernal_3=m_kernal_3,
        m_kernal_bias_3=m_kernal_bias_3,
        m_kernal_4=m_kernal_4,
        m_kernal_bias_4=m_kernal_bias_4,
        m_weights_input_hidden=m_weights_input_hidden,
        m_bias_hidden=m_bias_hidden,
        m_weights_hidden_hidden2=m_weights_hidden_hidden2,
        m_bias_hidden2=m_bias_hidden2,
        m_weights_hidden2_output=m_weights_hidden2_output,
        m_bias_output=m_bias_output,
    
        v_kernal=v_kernal,
        v_kernal_bias=v_kernal_bias,
        v_kernal_2=v_kernal_2,
        v_kernal_bias_2=v_kernal_bias_2,
        v_kernal_3=v_kernal_3,
        v_kernal_bias_3=v_kernal_bias_3,
        v_kernal_4=v_kernal_4,
        v_kernal_bias_4=v_kernal_bias_4,
        v_weights_input_hidden=v_weights_input_hidden,
        v_bias_hidden=v_bias_hidden,
        v_weights_hidden_hidden2=v_weights_hidden_hidden2,
        v_bias_hidden2=v_bias_hidden2,
        v_weights_hidden2_output=v_weights_hidden2_output,
        v_bias_output=v_bias_output,

        t=t
    )
#load model
model = np.load("CNN_mnist_model.npz")

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

m_kernal = np.asarray(model["m_kernal"], dtype=np.float32)
m_kernal_bias = np.asarray(model["m_kernal_bias"], dtype=np.float32)
m_kernal_2 = np.asarray(model["m_kernal_2"], dtype=np.float32)
m_kernal_bias_2 = np.asarray(model["m_kernal_bias_2"], dtype=np.float32)
m_kernal_3 = np.asarray(model["m_kernal_3"], dtype=np.float32)
m_kernal_bias_3 = np.asarray(model["m_kernal_bias_3"], dtype=np.float32)
m_kernal_4 = np.asarray(model["m_kernal_4"], dtype=np.float32)
m_kernal_bias_4 = np.asarray(model["m_kernal_bias_4"], dtype=np.float32)
m_weights_input_hidden = np.asarray(model["m_weights_input_hidden"], dtype=np.float32)
m_bias_hidden = np.asarray(model["m_bias_hidden"], dtype=np.float32)
m_weights_hidden_hidden2 = np.asarray(model["m_weights_hidden_hidden2"], dtype=np.float32)
m_bias_hidden2 = np.asarray(model["m_bias_hidden2"], dtype=np.float32)
m_weights_hidden2_output = np.asarray(model["m_weights_hidden2_output"], dtype=np.float32)
m_bias_output = np.asarray(model["m_bias_output"], dtype=np.float32)

v_kernal = np.asarray(model["v_kernal"], dtype=np.float32)
v_kernal_bias = np.asarray(model["v_kernal_bias"], dtype=np.float32)
v_kernal_2 = np.asarray(model["v_kernal_2"], dtype=np.float32)
v_kernal_bias_2 = np.asarray(model["v_kernal_bias_2"], dtype=np.float32)
v_kernal_3 = np.asarray(model["v_kernal_3"], dtype=np.float32)
v_kernal_bias_3 = np.asarray(model["v_kernal_bias_3"], dtype=np.float32)
v_kernal_4 = np.asarray(model["v_kernal_4"], dtype=np.float32)
v_kernal_bias_4 = np.asarray(model["v_kernal_bias_4"], dtype=np.float32)
v_weights_input_hidden = np.asarray(model["v_weights_input_hidden"], dtype=np.float32)
v_bias_hidden = np.asarray(model["v_bias_hidden"], dtype=np.float32)
v_weights_hidden_hidden2 = np.asarray(model["v_weights_hidden_hidden2"], dtype=np.float32)
v_bias_hidden2 = np.asarray(model["v_bias_hidden2"], dtype=np.float32)
v_weights_hidden2_output = np.asarray(model["v_weights_hidden2_output"], dtype=np.float32)
v_bias_output = np.asarray(model["v_bias_output"], dtype=np.float32)

t = int(model["t"]) #not an array

print("batch count:",t)

for e in range(10): #train for 5 epochs
    batch_start = 0 #start from first training images again
    np.random.shuffle(dataset) #randomly shuffle again

    for b in range(25000//batch_size): #integer division (25000/64 is not clean)

        if keyboard.is_pressed("s"): #if s key is saved
            print("saving model...")
            np.savez(
                "CNN_mnist_model.npz",
                kernal=kernal,
                kernal_bias=kernal_bias,
                kernal_2=kernal_2,
                kernal_bias_2=kernal_bias_2,
                kernal_3=kernal_3,
                kernal_bias_3=kernal_bias_3,
                kernal_4=kernal_4,
                kernal_bias_4=kernal_bias_4,
                weights_input_hidden=weights_input_hidden,
                bias_hidden=bias_hidden,
                weights_hidden_hidden2=weights_hidden_hidden2,
                bias_hidden2=bias_hidden2,
                weights_hidden2_output=weights_hidden2_output,
                bias_output=bias_output,

                m_kernal=m_kernal,
                m_kernal_bias=m_kernal_bias,
                m_kernal_2=m_kernal_2,
                m_kernal_bias_2=m_kernal_bias_2,
                m_kernal_3=m_kernal_3,
                m_kernal_bias_3=m_kernal_bias_3,
                m_kernal_4=m_kernal_4,
                m_kernal_bias_4=m_kernal_bias_4,
                m_weights_input_hidden=m_weights_input_hidden,
                m_bias_hidden=m_bias_hidden,
                m_weights_hidden_hidden2=m_weights_hidden_hidden2,
                m_bias_hidden2=m_bias_hidden2,
                m_weights_hidden2_output=m_weights_hidden2_output,
                m_bias_output=m_bias_output,
            
                v_kernal=v_kernal,
                v_kernal_bias=v_kernal_bias,
                v_kernal_2=v_kernal_2,
                v_kernal_bias_2=v_kernal_bias_2,
                v_kernal_3=v_kernal_3,
                v_kernal_bias_3=v_kernal_bias_3,
                v_kernal_4=v_kernal_4,
                v_kernal_bias_4=v_kernal_bias_4,
                v_weights_input_hidden=v_weights_input_hidden,
                v_bias_hidden=v_bias_hidden,
                v_weights_hidden_hidden2=v_weights_hidden_hidden2,
                v_bias_hidden2=v_bias_hidden2,
                v_weights_hidden2_output=v_weights_hidden2_output,
                v_bias_output=v_bias_output,

                t=t
            )
            print("model saved")
            
        batch_cost = 0
        batch_accuracy = []

        #all weights, bias and activations accross all trainig examples in batch

        predictions = []

        batch_training_images_data = []

        batch_pool_mask1 = []
        batch_pool_mask2 = []
        batch_pool_mask3 = []
        batch_pool_mask4 = []

        batch_activation_pool1 = []
        batch_raw_value_conv1 = []

        batch_activation_pool2 = []
        batch_raw_value_conv2 = []

        batch_activation_pool3 = []
        batch_raw_value_conv3 = []

        batch_activation_pool4 = []
        batch_raw_value_conv4 = []


        batch_activation_input = []

        batch_activation_hidden = []
        batch_weighted_sum_hidden = []

        batch_activation_hidden2 = []
        batch_weighted_sum_hidden2 = []

        batch_activation_output = []
        batch_weighted_sum_output = []

        batch_expected_output = []

        start_time = time.time()
        for i in range(batch_start, batch_start + batch_size):

            #<==========conv layer 1==========>

            file_name, training_labels_data = dataset[i]
            training_images_data = np.asarray( #convert to gpu
                np.load(f"Processed/{file_name}.npy")
            )
            batch_training_images_data.append(training_images_data)
            #add 1 layer of padding to each side of the image matrix (allows a 3x3 kernal to passover every pixel)
            padded_training_images_data = np.pad((training_images_data/255)-0.5, 1, mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)
            #divided by 255 to normilize in 0-1, then -0.5 to center around 0

            #10 outputs for filtered images (each the same shape of original image)
            region = sliding_window_view(padded_training_images_data, (3,3))
            #e.g if x=0, y=0 output matrix of 3x3 region with the top left corner centered at 0,0
            region = region[None, :, :, :, :] #adjusting shapes to broadcast together
            kernals = kernal[:, None, None, :, :]

            #multiply each corosponding element in the matrices(kernal and region) then sum output matrix, then add kernal bias
            value = np.sum(region * kernals, axis=(3,4)) #sum accross axis 3 and 4 (3x3)
            raw_conv1 = value + kernal_bias[:, None, None] #adjusting shapes to broadcast correctly and add kernal bias accross all feature maps
            output = ReLU(raw_conv1) #value for filtered pixel (convoloution of original image and kernal) + activation function
            batch_raw_value_conv1.append(raw_conv1)

            feature_maps, pool_mask = maxpool2x2(output) #max pooling 2x2 function (vectorised)
            batch_activation_pool1.append(feature_maps)
            batch_pool_mask1.append(pool_mask)

            #<==========conv layer 2==========>

            #add 1 layer of padding to each side of the image matrix (not the vector of image matricies) (allows a 3x3 kernal to passover every pixel)
            padded_previous_feature_maps = np.pad(feature_maps, ((0,0), (1,1), (1,1)), mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)

            #20 outputs for filtered images (each the same shape of original image)
            region = sliding_window_view(padded_previous_feature_maps, (3,3), axis=(1,2))
            #e.g if x=0, y=0 output matrix of 3x3 region with the top left corner centered at 0,0
            region = region[None, :, :, :, :, :] #adjusting shapes to broadcast correctly
            kernals = kernal_2[:, :, None, None, :, :]

            #multiply each corosponding element in the matrices(kernal and region) then sum output matrix, then add kernal bias
            raw_conv2 = np.sum(region * kernals, axis=(1,4,5)) + kernal_bias_2[:, None, None] #sum accross feature maps amd kernal values then add kernal bias accross all feature maps
            output_2 = ReLU(raw_conv2) #sum of values for filtered pixels across (convoloution of original image and kernal)
            batch_raw_value_conv2.append(raw_conv2)

            feature_maps_2, pool2_mask = maxpool2x2(output_2) #max pooling 2x2 function (vectorised)
            batch_activation_pool2.append(feature_maps_2)
            batch_pool_mask2.append(pool2_mask)

            #<==========conv layer 3==========>

            #add 1 layer of padding to each side of the image matrix (not the vector of image matricies) (allows a 3x3 kernal to passover every pixel)
            padded_previous_feature_maps = np.pad(feature_maps_2, ((0,0), (1,1), (1,1)), mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)

            #40 outputs for filtered images (each the same shape of original image)
            region = sliding_window_view(padded_previous_feature_maps, (3,3), axis=(1,2))
            #e.g if x=0, y=0 output matrix of 3x3 region with the top left corner centered at 0,0
            region = region[None, :, :, :, :, :] #adjusting shapes to broadcast correctly
            kernals = kernal_3[:, :, None, None, :, :]

            #multiply each corosponding element in the matrices(kernal and region) then sum output matrix, then add kernal bias
            raw_conv3 = np.sum(region * kernals, axis=(1,4,5)) + kernal_bias_3[:, None, None] #sum accross feature maps amd kernal values then add kernal bias accross all feature maps
            output_3 = ReLU(raw_conv3) #sum of values for filtered pixels across (convoloution of original image and kernal)
            batch_raw_value_conv3.append(raw_conv3)

            feature_maps_3, pool3_mask = maxpool2x2(output_3) #max pooling 2x2 function (vectorised)
            batch_activation_pool3.append(feature_maps_3)
            batch_pool_mask3.append(pool3_mask)

            #<==========conv layer 4==========>

            #add 1 layer of padding to each side of the image matrix (not the vector of image matricies) (allows a 3x3 kernal to passover every pixel)
            padded_previous_feature_maps = np.pad(feature_maps_3, ((0,0), (1,1), (1,1)), mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)

            #80 outputs for filtered images (each the same shape of original image)
            region = sliding_window_view(padded_previous_feature_maps, (3,3), axis=(1,2))
            #e.g if x=0, y=0 output matrix of 3x3 region with the top left corner centered at 0,0
            region = region[None, :, :, :, :, :] #adjusting shapes to broadcast correctly
            kernals = kernal_4[:, :, None, None, :, :]

            #multiply each corosponding element in the matrices(kernal and region) then sum output matrix, then add kernal bias
            raw_conv4 = np.sum(region * kernals, axis=(1,4,5)) + kernal_bias_4[:, None, None] #sum accross feature maps amd kernal values then add kernal bias accross all feature maps
            output_4 = ReLU(raw_conv4) #sum of values for filtered pixels across (convoloution of original image and kernal)
            batch_raw_value_conv4.append(raw_conv4)

            feature_maps_4, pool4_mask = maxpool2x2(output_4) #max pooling 2x2 function (vectorised)
            batch_activation_pool4.append(feature_maps_4)
            batch_pool_mask4.append(pool4_mask)

            input_layer = feature_maps_4.flatten()

            #<==========dense layers==========>

            #input data (18000 neurons)
            #flatten converts 80x15x15 matrix of grayscale values down to 1d 18000 long vector of grayscale values
            #scaled down grayscale values from 255 to 0-1 range (activation range)
            inputs = input_layer
            batch_activation_input.append(inputs)

            #activations

            #hidden layer values without ReLU
            hidden_raw = np.dot(inputs, weights_input_hidden) + bias_hidden
            batch_weighted_sum_hidden.append(hidden_raw) #recording hidden layer weighted sum for training example
            hidden_activation = ReLU(hidden_raw) #hidden layer activations
            batch_activation_hidden.append(hidden_activation)

            #second hidden layer values without ReLU
            hidden2_raw = np.dot(hidden_activation, weights_hidden_hidden2) + bias_hidden2
            batch_weighted_sum_hidden2.append(hidden2_raw) #recording hidden2 layer weighted sum for training example
            hidden2_activation = ReLU(hidden2_raw) #second hidden layer activations
            batch_activation_hidden2.append(hidden2_activation)

            #output layer values without ReLU
            output_raw = np.dot(hidden2_activation, weights_hidden2_output) + bias_output
            batch_weighted_sum_output.append(output_raw) #recording output layer weighted sum for training example
            output_activation = softmax(output_raw) #output layer activations
            batch_activation_output.append(output_activation)

            cost = cross_entropy_cost(output_activation, training_labels_data)
            accuracy = accurate(output_activation, training_labels_data)
            predictions.append(np.argmax(output_activation))

            batch_cost += cost
            batch_accuracy.append(accuracy)

        batch_cost = batch_cost / batch_size
        print("cost:", batch_cost)
        batch_accuracy = np.mean(batch_accuracy)
        print("accuracy:", batch_accuracy)
        print("output neurons:", output_activation)
        print("predicted cats:", predictions.count(1))
        print("predicted dogs:", predictions.count(0))

        #backprop

        #first convolutional layer
        total_dkw1 = np.zeros_like(kernal)
        total_dkb1 = np.zeros_like(kernal_bias)

        #second convolutional layer
        total_dkw2 = np.zeros_like(kernal_2)
        total_dkb2 = np.zeros_like(kernal_bias_2)

        #third convolutional layer
        total_dkw3 = np.zeros_like(kernal_3)
        total_dkb3 = np.zeros_like(kernal_bias_3)

        #fourth convolutional layer
        total_dkw4 = np.zeros_like(kernal_4)
        total_dkb4 = np.zeros_like(kernal_bias_4)

        #hidden layer (first layer)
        total_dw1 = np.zeros_like(weights_hidden2_output) #same shape as weight matrix but all values at 0 for start
        total_db1 = np.zeros_like(bias_output) #same shape as bias array but alll values at 0 for start

        #hidden2 layer (second layer)
        total_dw2 = np.zeros_like(weights_hidden_hidden2)
        total_db2 = np.zeros_like(bias_hidden2)

        #output layer (third layer)
        total_dw3 = np.zeros_like(weights_input_hidden)
        total_db3 = np.zeros_like(bias_hidden)

        for i in range(batch_size):
            training_images_data = batch_training_images_data[i]

            #z=weighted sum, a=activation, y=expected/target
            z1 = batch_weighted_sum_output[i]
            a1 = batch_activation_output[i]

            z2 = batch_weighted_sum_hidden2[i]
            a2 = batch_activation_hidden2[i]

            z3 = batch_weighted_sum_hidden[i]
            a3 = batch_activation_hidden[i]

            a4 = batch_activation_input[i]

            a5 = batch_activation_pool4[i]
            a6 = batch_raw_value_conv4[i]

            a7 = batch_activation_pool3[i]
            a8 = batch_raw_value_conv3[i]
            
            a9 = batch_activation_pool2[i]
            a10 = batch_raw_value_conv2[i]

            a11 = batch_activation_pool1[i]
            a12 = batch_raw_value_conv1[i]

            #how sensitive pool1 layer raw values (before ReLU) are to convo1 layer values - 1 for maximum value in pooling kernal window, 0 for others (m1)
            m1 = batch_pool_mask1[i] #partial derivative of pool1 with respect to convo1

            #how sensitive pool2 layer raw values (before ReLU) are to covo2 layer values - 1 for maximum value in pooling kernal window, 0 for others (m2)
            m2 = batch_pool_mask2[i] #partial derivative of pool2 with respect to convo2

            #how sensitive pool3 layer raw values (before ReLU) are to convo3 layer values - 1 for maximum value in pooling kernal window, 0 for others (m3)
            m3 = batch_pool_mask3[i] #partial derivative of pool3 with respect to convo3
            
            #how sensitive pool4 layer raw values (before ReLU) are to covo4 layer values - 1 for maximum value in pooling kernal window, 0 for others (m4)
            m4 = batch_pool_mask4[i] #partial derivative of pool4 with respect to convo4

            y = batch_expected_output[i]


        #calculating partial derivatives of cost with respect to every weight and bias accross all training examples in the batch

        #<==========dense layers backprop==========>
            #how sensitive the cost function is to the output layer weighted sums
            #for softmax combined with cross entropy this simplifies to:
            #predicted probabilities - expected probabilities
            delta1 =  a1 - y #how sensitive the cost function is to the output layer weighted sum (derived from softmax and cross entropy)

        #64 collumns -(64hidden2 neurons-64 activations), to
        #2 rows -(2output neurons-2 cost derivatives)
        #for every location in the weight matrix, calculate its derivative from previous corosponding activation (activation it was multiplied by)
        #and cost functions sensitivity to output layer weighted sum and multiply them together (using outer product)
            total_dw1 += np.outer(a2, delta1)

        #2 collumns -(2 output neurons)
        #cost functions sensitivity to output layer weighted sum
        #and how sensitive weighted sum is to the bias (1)
            total_db1 += delta1

        #64 collumns -(64hidden2 neurons), to
        #2 rows -(2output neurons)
            delta2 = weights_hidden2_output @ delta1 #how sensitive output layer weighted sum is to previous activation * delta1 (matrix multiplication)
            delta2 *= ReLU_derivative(z2) #how sensitive hidden2 layer activations are to hidden2 layer weighted sum

        #128 collumns -(128hidden neurons-128 activations), to
        #64 rows -(64hidden2 neurons-64 cost derivatives)
        #for every location in the weight matrix, calculate its derivative from previous corosponding activation (activation it was multiplied by)
        #and cost functions sensitivity to hidden2 layer weighted sum  and multiply them together (using outer product)
            total_dw2 += np.outer(a3, delta2)

        #64 collumns -(64 hidden2 neurons)
        #cost functions sensitivity to hidden2 layer weighted sum
        #and how sensitive weighted sum is to the bias (1)
            total_db2 += delta2

        #128 collumns -(128hidden neurons), to
        #64 rows -(64hidden2 neurons-64 cost derivatives)
            delta3 = weights_hidden_hidden2 @ delta2#how sensitive hidden2 layer weighted sum is to previous activation * delta2 (matrix multiplication)
            delta3 *= ReLU_derivative(z3) #how sensitive hidden layer activations are to hidden layer weighted sum

        #18000 collumns -(18000input neurons-18000 activations), to
        #128 rows -(128hidden neurons-128 cost derivatives)
        #for every location in the weight matrix, calculate its derivative from previous corosponding activation (activation it was multiplied by)
        #and cost functions sensitivity to hidden layer weighted sum  and multiply them together (using outer product)
            total_dw3 += np.outer(a4, delta3)

        #128 collumns -(128 hidden neurons)
        #cost functions sensitivity to hidden layer weighted sum
        #and how sensitive weighted sum is to the bias (1)
            total_db3 += delta3

        
        #<==========conv layer 4 backprop==========>

        #18000 collumns -(18000input neurons), to
        #128 rows -(128hidden neurons)
            delta4 = weights_input_hidden @ delta3 #how sensitive hidden layer weighted sum is to previous activation * delta3
            delta4 = delta4.reshape(80, 15, 15) #how sensitive cost is to global average pooled values

            grad = delta4[:, :, :, None, None] #reshaping delta4 to broadcast with m4 regions
            #expand the pooled gradient back into convolution layer 4 space by assigning each delta4
            #value to the maximum location in its 2x2 pooling region using the pooling mask (m4)
            m4_regions = sliding_window_view(m4, (2,2), axis=(1,2))
            m4_regions = m4_regions[:, ::2, ::2] #remove every 2nd region to avoid overlap (not convolution)
            dconv4 = grad * m4_regions #how sensitive the cost is to each convolutional value
            dconv4 = dconv4.transpose(0, 1, 3, 2, 4).reshape(80, 30, 30) #convert output of grad * m4 regions (80,15,2,15,2) into (80,15,15,2,2) then into (80,30,30)
            #no data is added only representing it in diffrent shapes, e.g. instead of for every x have a x 2xregion and every y have a y 2xregion have for every y and x have a 2x2 region
            #then represent 15x15 2x2 regions as just 30x30 locations
            #apply simgoid derivative to calculate sensitivity to raw convo4 values
            dconv4 *= ReLU_derivative(a6)

            #how sensitive convo4 layer values are to kernal4 and kernal4 bias - the value multiplied against kernal value (corosponding previous activation) and 1 for bias
            padded_previous_feature_maps = np.pad(a7, ((0,0), (1,1), (1,1)), mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)
            regions = sliding_window_view(padded_previous_feature_maps, (3,3), axis=(1,2)) #select 3x3 kernals from every 30x30 position in (40,30,30)-using function, (actually32x32 for padding)
            #adjust shapes to allow dconv4 and regions to broadcast correctly (element wise mutliplication)
            dconv4_b = dconv4[:, None, :, :, None, None]
            regions_b = regions[None, :, :, :, :, :]
            total = dconv4_b * regions_b #element wise mutliplication
            total_dkw4 += total.sum(axis=(2,3)) #sum over postions

            total_dkb4 += dconv4.sum(axis=(1,2)) #how sensitive convoloutional values are to bias is just 1 (sum dconv2 accross postions 30,30 to leave 80)

        #<==========conv layer 3 backprop==========>

            delta5_padded = np.zeros((40,32,32)) #padding used to record convolutions
#how sensitive convo4 is to pool3 can be represented by the kernal weights in the positions of corosponding pool3 values, hence delta5[k, x:x+3, y:y+3] and kernal_4[j, k]
#then multiply by how sensitive cost is to covo2 layer activations
            grad = dconv4[:, :, :, None, None, None] #adjusting shapes for correct broadcast
            kernals = kernal_4[:, None, None, :, :, :]
            patches = (grad * kernals).sum(axis=(0)).transpose(2, 0, 1, 3, 4) #shape from 80 30 30 40 3 3 to 30 30 40 3 3 to 40 30 30 3 3
#overlapping 3x3 regions of gradients are added on top of eachother
#becasue covo3 layer values are more affected by pool values closer to the center of their feature map (because they are convoluted more often)
#so the derivatives are added on top of eachother due to the nature of convolution
            for x in range(30): #scatter add, unavoidable x,y loop - no numpy vectorisation available, but still about 200x faster
                for y in range(30):
                    delta5_padded[:, x:x+3, y:y+3] += patches[:, x, y] #shape 40 32 32
            delta5 = delta5_padded[:, 1:-1, 1:-1] #trimm off padding used to record convolutions, shape 40 30 30


            grad = delta5[:, :, :, None, None] #reshaping delta5 to broadcast with m3 regions
            #expand the pooled gradient back into convolution layer 3 space by assigning each delta5
            #value to the maximum location in its 2x2 pooling region using the pooling mask (m3)
            m3_regions = sliding_window_view(m3, (2,2), axis=(1,2))
            m3_regions = m3_regions[:, ::2, ::2] #remove every 2nd region to avoid overlap (not convolution)
            dconv3 = grad * m3_regions #how sensitive the cost is to each convolutional value
            dconv3 = dconv3.transpose(0, 1, 3, 2, 4).reshape(40, 60, 60) #convert output of grad * m3 regions (40,30,2,30,2) into (40,30,30,2,2) then into (40,60,60)
            #no data is added only representing it in diffrent shapes, e.g. instead of for every x have a x 2xregion and every y have a y 2xregion have for every y and x have a 2x2 region
            #then represent 30x30 2x2 regions as just 60x60 locations
            #apply simgoid derivative to calculate sensitivity to raw convo3 values
            dconv3 *= ReLU_derivative(a8)

            #how sensitive convo3 layer values are to kernal3 and kernal3 bias - the value multiplied against kernal value (corosponding previous activation) and 1 for bias
            padded_previous_feature_maps = np.pad(a9, ((0,0), (1,1), (1,1)), mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)
            regions = sliding_window_view(padded_previous_feature_maps, (3,3), axis=(1,2)) #select 3x3 kernals from every 60x60 position in (20,60,60)-using function, (actually62x62 for padding)
            #adjust shapes to allow dconv3 and regions to broadcast correctly (element wise mutliplication)
            dconv3_b = dconv3[:, None, :, :, None, None]
            regions_b = regions[None, :, :, :, :, :]
            total = dconv3_b * regions_b #element wise mutliplication
            total_dkw3 += total.sum(axis=(2,3)) #sum over postions

            total_dkb3 += dconv3.sum(axis=(1,2)) #how sensitive convoloutional values are to bias is just 1 (sum dconv3 accross postions 60,60 to leave 40)

        #<==========conv layer 2 backprop==========>

            delta6_padded = np.zeros((20,62,62)) #padding used to record convolutions
#how sensitive convo3 is to pool2 can be represented by the kernal weights in the positions of corosponding pool2 values, hence delta6[k, x:x+3, y:y+3] and kernal_3[j, k]
#then multiply by how sensitive cost is to covo2 layer activations
            grad = dconv3[:, :, :, None, None, None] #adjusting shapes for correct broadcast
            kernals = kernal_3[:, None, None, :, :, :]
            patches = (grad * kernals).sum(axis=(0)).transpose(2, 0, 1, 3, 4) #shape from 40 60 60 20 3 3 to 60 60 20 3 3 to 20 60 60 3 3
#overlapping 3x3 regions of gradients are added on top of eachother
#becasue covo3 layer values are more affected by pool values closer to the center of their feature map (because they are convoluted more often)
#so the derivatives are added on top of eachother due to the nature of convolution
            for x in range(60): #scatter add, unavoidable x,y loop - no numpy vectorisation available, but still about 200x faster
                for y in range(60):
                    delta6_padded[:, x:x+3, y:y+3] += patches[:, x, y] #shape 20 62 62
            delta6 = delta6_padded[:, 1:-1, 1:-1] #trimm off padding used to record convolutions, shape 20 60 60


            grad = delta6[:, :, :, None, None] #reshaping delta6 to broadcast with m2 regions
            #expand the pooled gradient back into convolution layer 2 space by assigning each delta6
            #value to the maximum location in its 2x2 pooling region using the pooling mask (m2)
            m2_regions = sliding_window_view(m2, (2,2), axis=(1,2))
            m2_regions = m2_regions[:, ::2, ::2] #remove every 2nd region to avoid overlap (not convolution)
            dconv2 = grad * m2_regions #how sensitive the cost is to each convolutional value
            dconv2 = dconv2.transpose(0, 1, 3, 2, 4).reshape(20, 120, 120) #convert output of grad * m2 regions (20,60,2,60,2) into (20,60,60,2,2) then into (20,120,120)
            #no data is added only representing it in diffrent shapes, e.g. instead of for every x have a x 2xregion and every y have a y 2xregion have for every y and x have a 2x2 region
            #then represent 60x60 2x2 regions as just 120x120 locations
            #apply simgoid derivative to calculate sensitivity to raw convo2 values
            dconv2 *= ReLU_derivative(a10)

            #how sensitive convo2 layer values are to kernal2 and kernal2 bias - the value multiplied against kernal value (corosponding previous activation) and 1 for bias
            padded_previous_feature_maps = np.pad(a11, ((0,0), (1,1), (1,1)), mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)
            regions = sliding_window_view(padded_previous_feature_maps, (3,3), axis=(1,2)) #select 3x3 kernals from every 120x120 position in (10,120,120)-using function, (actually122x122 for padding)
            #adjust shapes to allow dcov2 and regions to broadcast correctly (element wise mutliplication)
            dconv2_b = dconv2[:, None, :, :, None, None]
            regions_b = regions[None, :, :, :, :, :]
            total = dconv2_b * regions_b #element wise mutliplication
            total_dkw2 += total.sum(axis=(2,3)) #sum over postions

            total_dkb2 += dconv2.sum(axis=(1,2)) #how sensitive convoloutional values are to bias is just 1 (sum dconv2 accross postions 120,120 to leave 20)

        #<==========conv layer 1 backprop==========>

            delta7_padded = np.zeros((10,122,122)) #padding used to record convolutions
#how sensitive convo2 is to pool1 can be represented by the kernal weights in the positions of corosponding pool1 values, hence delta7[k, x:x+3, y:y+3] and kernal_2[j, k]
#then multiply by how sensitive cost is to covo2 layer activations
            grad = dconv2[:, :, :, None, None, None] #adjusting shapes for correct broadcast
            kernals = kernal_2[:, None, None, :, :, :]
            patches = (grad * kernals).sum(axis=(0)).transpose(2, 0, 1, 3, 4) #shape from 20 120 120 10 3 3 to 120 120 10 3 3 to 10 120 120 3 3
#overlapping 3x3 regions of gradients are added on top of eachother
#becasue covo2 layer values are more affected by pool values closer to the center of their feature map (because they are convoluted more often)
#so the derivatives are added on top of eachother due to the nature of convolution
            for x in range(120): #scatter add, unavoidable x,y loop - no numpy vectorisation available, but still about 200x faster
                for y in range(120):
                    delta7_padded[:, x:x+3, y:y+3] += patches[:, x, y] #shape 10 122 122
            delta7 = delta7_padded[:, 1:-1, 1:-1] #trimm off padding used to record convolutions, shape 10 120 120

            grad = delta7[:, :, :, None, None] #reshaping delta7 to broadcast with m1 regions
            #expand the pooled gradient back into convolution layer 1 space by assigning each delta7
            #value to the maximum location in its 2x2 pooling region using the pooling mask (m1)
            m1_regions = sliding_window_view(m1, (2,2), axis=(1,2))
            m1_regions = m1_regions[:, ::2, ::2] #remove every 2nd region to avoid overlap (not convolution)
            dconv1 = grad * m1_regions #how sensitive the cost is to each convolutional value
            dconv1 = dconv1.transpose(0, 1, 3, 2, 4).reshape(10, 240, 240) #convert output of grad * m1 regions (10,120,2,120,2) into (10,120,120,2,2) then into (10,240,240)
            #no data is added only representing it in diffrent shapes, e.g. instead of for every x have a x 2xregion and every y have a y 2xregion have for every y and x have a 2x2 region
            #then represent 120x120 2x2 regions as just 240x240 locations
            #apply simgoid derivative to calculate sensitivity to raw convo2 values
            dconv1 *= ReLU_derivative(a12)

            padded_previous_feature_maps = np.pad((training_images_data/255)-0.5, ((1,1), (1,1)), mode="constant", constant_values=0) #values for pading layers is a constant(0) (only affects last 2)
            regions = sliding_window_view(padded_previous_feature_maps, (3,3), axis=(0, 1)) #select 3x3 kernals from every 240x240 position in (240,240)-using function, (actually242x242 for padding)
            #adjust shapes to allow dconv1 and regions to broadcast correctly (element wise mutliplication)
            dconv1_b = dconv1[:, :, :, None, None]
            regions_b = regions[None, :, :, :, :]
            total = dconv1_b * regions_b #element wise mutliplication
            total_dkw1 += total.sum(axis=(1,2)) #sum over postions

            total_dkb1 += dconv1.sum(axis=(1,2)) #how sensitive convoloutional values are to bias is just 1 (sum dconv1 accross postions 240,240 to leave 10)

        #avarage gradients accros batch
        dkw1 = total_dkw1 / batch_size
        dkb1 = total_dkb1 / batch_size

        dkw2 = total_dkw2 / batch_size
        dkb2 = total_dkb2 / batch_size

        dkw3 = total_dkw3 / batch_size
        dkb3 = total_dkb3 / batch_size

        dkw4 = total_dkw4 / batch_size
        dkb4 = total_dkb4 / batch_size

        dw1 = total_dw1 / batch_size
        db1 = total_db1 / batch_size

        dw2 = total_dw2 / batch_size
        db2 = total_db2 / batch_size

        dw3 = total_dw3 / batch_size
        db3 = total_db3 / batch_size

        t += 1 #increase number of previous gradients

        params = [
            kernal,
            kernal_bias,
            kernal_2,
            kernal_bias_2,
            kernal_3,
            kernal_bias_3,
            kernal_4,
            kernal_bias_4,
            weights_input_hidden,
            bias_hidden,
            weights_hidden_hidden2,
            bias_hidden2,
            weights_hidden2_output,
            bias_output,
        ]
        grads = [
            dkw1,
            dkb1,
            dkw2,
            dkb2,
            dkw3,
            dkb3,
            dkw4,
            dkb4,
            dw3,
            db3,
            dw2,
            db2,
            dw1,
            db1
        ]
        m = [
            m_kernal,
            m_kernal_bias,
            m_kernal_2,
            m_kernal_bias_2,
            m_kernal_3,
            m_kernal_bias_3,
            m_kernal_4,
            m_kernal_bias_4,
            m_weights_input_hidden,
            m_bias_hidden,
            m_weights_hidden_hidden2,
            m_bias_hidden2,
            m_weights_hidden2_output,
            m_bias_output,
        ]
        v = [
            v_kernal,
            v_kernal_bias,
            v_kernal_2,
            v_kernal_bias_2,
            v_kernal_3,
            v_kernal_bias_3,
            v_kernal_4,
            v_kernal_bias_4,
            v_weights_input_hidden,
            v_bias_hidden,
            v_weights_hidden_hidden2,
            v_bias_hidden2,
            v_weights_hidden2_output,
            v_bias_output,
        ]

        #gradient decent via adam optimizer
        params, m, v = update_adam(params, grads, m, v, t)

        #gradient decent

        batch_start += batch_size #next batch

        if b % 10 == 0: #save model every ten batches
            np.savez(
                "CNN_mnist_model.npz",
                kernal=params[0],
                kernal_bias=params[1],
                kernal_2=params[2],
                kernal_bias_2=params[3],
                kernal_3=params[4],
                kernal_bias_3=params[5],
                kernal_4=params[6],
                kernal_bias_4=params[7],
                weights_input_hidden=params[8],
                bias_hidden=params[9],
                weights_hidden_hidden2=params[10],
                bias_hidden2=params[11],
                weights_hidden2_output=params[12],
                bias_output=params[13],

                m_kernal=m[0],
                m_kernal_bias=m[1],
                m_kernal_2=m[2],
                m_kernal_bias_2=m[3],
                m_kernal_3=m[4],
                m_kernal_bias_3=m[5],
                m_kernal_4=m[6],
                m_kernal_bias_4=m[7],
                m_weights_input_hidden=m[8],
                m_bias_hidden=m[9],
                m_weights_hidden_hidden2=m[10],
                m_bias_hidden2=m[11],
                m_weights_hidden2_output=m[12],
                m_bias_output=m[13],
            
                v_kernal=v[0],
                v_kernal_bias=v[1],
                v_kernal_2=v[2],
                v_kernal_bias_2=v[3],
                v_kernal_3=v[4],
                v_kernal_bias_3=v[5],
                v_kernal_4=v[6],
                v_kernal_bias_4=v[7],
                v_weights_input_hidden=v[8],
                v_bias_hidden=v[9],
                v_weights_hidden_hidden2=v[10],
                v_bias_hidden2=v[11],
                v_weights_hidden2_output=v[12],
                v_bias_output=v[13],

                t=t
            )
            print("model saved")

        end_time = time.time()
        print(f"{end_time - start_time} seconds")

        print("-------------------batch over-------------------")
    
    print("epoch:",e+1, " complete")
    np.savez( #save model after every epoch (all batches completed)
        "CNN_mnist_model.npz",
        kernal=params[0],
        kernal_bias=params[1],
        kernal_2=params[2],
        kernal_bias_2=params[3],
        kernal_3=params[4],
        kernal_bias_3=params[5],
        kernal_4=params[6],
        kernal_bias_4=params[7],
        weights_input_hidden=params[8],
        bias_hidden=params[9],
        weights_hidden_hidden2=params[10],
        bias_hidden2=params[11],
        weights_hidden2_output=params[12],
        bias_output=params[13],

        m_kernal=m[0],
        m_kernal_bias=m[1],
        m_kernal_2=m[2],
        m_kernal_bias_2=m[3],
        m_kernal_3=m[4],
        m_kernal_bias_3=m[5],
        m_kernal_4=m[6],
        m_kernal_bias_4=m[7],
        m_weights_input_hidden=m[8],
        m_bias_hidden=m[9],
        m_weights_hidden_hidden2=m[10],
        m_bias_hidden2=m[11],
        m_weights_hidden2_output=m[12],
        m_bias_output=m[13],
    
        v_kernal=v[0],
        v_kernal_bias=v[1],
        v_kernal_2=v[2],
        v_kernal_bias_2=v[3],
        v_kernal_3=v[4],
        v_kernal_bias_3=v[5],
        v_kernal_4=v[6],
        v_kernal_bias_4=v[7],
        v_weights_input_hidden=v[8],
        v_bias_hidden=v[9],
        v_weights_hidden_hidden2=v[10],
        v_bias_hidden2=v[11],
        v_weights_hidden2_output=v[12],
        v_bias_output=v[13],

        t=t
    )
    print("model saved")
print("training finished")

#adapt for cat+dog dataset

#adapt for object detection (YOLO)

#adapt for self supervised object detection (like dino v3 - simmiler to how brain learns - no labels)