import socket #pythons built in socket libary - allows for communicating over network using protocols like TCP
import threading #pythons built in threading libary - allows for execution of many tasks within a process e.g. many sockets
import numpy as np #for recieving and sending np data aswell as matrix manipulation
import pickle #encoding and decoding np (+other) data (built in)
import os #check for existing model and load dataset files
import time

files = os.listdir("Processed")
np.random.shuffle(files)

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

def get_model():
    return {
        "kernal": kernal,
        "kernal_bias": kernal_bias,
        "kernal_2": kernal_2,
        "kernal_bias_2": kernal_bias_2,
        "kernal_3": kernal_3,
        "kernal_bias_3": kernal_bias_3,
        "kernal_4": kernal_4,
        "kernal_bias_4": kernal_bias_4,
        "weights_input_hidden": weights_input_hidden,
        "bias_hidden": bias_hidden,
        "weights_hidden_hidden2": weights_hidden_hidden2,
        "bias_hidden2": bias_hidden2,
        "weights_hidden2_output": weights_hidden2_output,
        "bias_output": bias_output,
    }

batch_size = 32

learning_rate = 0.002

clients = [] #list of connected clients

WAITING = 0
CALIBRATING = 1
TRAINING = 2

state = WAITING

start_time = None

recalibrate = True

batch_complete_addition = 0

#update adam optimizer (takes in the model params, current gradients, m(gradients direction), v(gradient magnitude) and t(number of previous gradients))
def update_adam(params, grads, m, v, t, learning_rate=0.002, beta1=0.9, beta2=0.99, epsilon=1e-8): #epsilon to prevent dividing by 0

    gradient_names = [ #gradient names list to work with the averaged gradeints dictionary instead of the expected list input
        "kernal",
        "kernal_bias",
        "kernal_2",
        "kernal_bias_2",
        "kernal_3",
        "kernal_bias_3",
        "kernal_4",
        "kernal_bias_4",
        "weights_input_hidden",
        "bias_hidden",
        "weights_hidden_hidden2",
        "bias_hidden2",
        "weights_hidden2_output",
        "bias_output",
    ]

    for i in range(len(params)): #for every model parameter

        grad = grads[gradient_names[i]]


        #update moving weighted averages
        m[i] = beta1 * m[i] + (1-beta1) * grad #weighted average of previous weighted average and current gradient
        v[i] = beta2 * v[i] + (1-beta2) * (grad ** 2) #uses diffrent beta to compensate for squaring

        #bias correction
        #when computing first gradients the moving average is smaller than it should be becuse the previous gradients no longer exist so are 0
        #so this boosts gradients at the start while resolving to normal as number of gradients increases
        corrected_m = m[i] / (1-beta1 **t)
        corrected_v = v[i] / (1-beta2 **t)

        #scales parameters individually and * LR
        params[i] -= learning_rate *  corrected_m / (np.sqrt(corrected_v) + epsilon) #epsilon to prevent dividing by 0

    return params, m, v

def send_object(socket, data):
    try:
        encoded = pickle.dumps(data) #encoding data usign pickle
        size = len(encoded) #size of encoded data (bytes)

        #used later to define the start and end of a object
        socket.sendall(size.to_bytes(8, "big")) #send size message first (8 bytes) -using big-endian byte order (most significant byte first), hence "big"
        #send actual data
        socket.sendall(encoded)
        return True

    except (ConnectionError, BrokenPipeError, OSError):
        return False

def receive_object(socket):
    #recieve size (using standard fucntion)
    size_data = recv_all(socket, 8) #recieve first 8 bytes (the size)

    if size_data is None:
        return None
    size = int.from_bytes(size_data, "big") #converts back into integer -using the same big-endian byte order (most significant byte first), hence "big"

    #receive exactly the number of bytes dictated by size, (the whole object) - (using standard function)
    encoded = recv_all(socket, size)

    if encoded is None:
        return None
    try:
        return pickle.loads(encoded) #decode data
    except Exception:
        return None
def recv_all(socket, size):
    data = b"" #byte variable type definition

    try:
        while len(data) < size: #while length of data currently retrieved is less than expected data size (continue untill all data is present)
            packet = socket.recv(size - len(data)) #recieve remaining length of data

            if not packet:
                print("socket", socket, "closed")
                return
            data += packet #add the packet of retrieved data to total data

        return data #return collected data
    
    except (ConnectionError, OSError):
        return None

def compute_gradients(gradients, start_time):
    global kernal
    global kernal_bias
    global kernal_2
    global kernal_bias_2
    global kernal_3
    global kernal_bias_3
    global kernal_4
    global kernal_bias_4
    global weights_input_hidden
    global bias_hidden
    global weights_hidden_hidden2
    global bias_hidden2
    global weights_hidden2_output
    global bias_output
    global batch_complete_addition
    global t

    total = {} #summed total of each parameter (sum accross clients summed gradients)

    for param in gradients[0]: #for every parameter (first summed gradient as a reference)
        total[param] = sum(gradient[param] for gradient in gradients) #sum param with that param from every gradient
    
    averaged_gradients = {}

    for param in total: #averaged gradients (divided accross all training examples)
        averaged_gradients[param] = (total[param] / batch_size)

    t+=1

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

    params, m, v = update_adam(params, averaged_gradients, m, v, t)

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
    if batch_complete_addition + batch_size >= len(files): #reset after every epoch
        batch_complete_addition = 0
    else:
        batch_complete_addition += batch_size
    print("batch complete")
    end_time = time.time()
    print("time:" + str(end_time - start_time) + "seconds")

#benchmark each clients speed, in order to decide how many training examples to give to each client,
def calibrate_workload(batch_size): #to avoid one client waiting on another to finish processing
    if not clients: #return nothing (cancel) if no clients to calibrate workload for
        return
    total_speed = 0
    speeds = []
    times = []
    total_allocations = 0
    for client in clients[:]:  #copy of list so we can remove safely
        if not send_object(client["socket"], "CALIBRATION"):
            client["connected"] = False
            clients.remove(client)
            continue
        #send current model and selected labels and images
        send_object(client["socket"], get_model())
        selected_files = files[0:0+batch_size] #all filenames within window
        selected_data = []
        selected_labels = []
        for filename in selected_files:
            data = np.load(os.path.join("processed", filename))
            selected_data.append(data) #np data from each file
        for file in selected_files: #giving label based on cat or dog status
            if file[0] == "C":
                selected_labels.append(1)
            if file[0] == "D":
                selected_labels.append(0)
        send_object(client["socket"], selected_data)
        send_object(client["socket"], selected_labels)
        time_taken = receive_object(client["socket"]) #recieve time taken to complete (used as benchmark)

        if time_taken is None:
            client["connected"] = False
            print("client disconnected:", client["address"])
            clients.remove(client)
            continue
        times.append(time_taken)
        if not clients:
            return

    for i in range(len(clients)):
        speed = batch_size/times[i]
        speeds.append(speed)
    total_speed = sum(speeds)
    for i in range(len(clients)):
        contrubution = speeds[i] / total_speed
        client = clients[i]
        client["training_examples_num"] = int(contrubution*batch_size)
        total_allocations += client["training_examples_num"]
    fastest_clients = np.argsort(times) #orders clients in time order (fastest first (lowest time first)) - (sorts indexs of time array)
    remainder = batch_size - total_allocations #if there are any training examples left un-allocated (due to truncation)

    if remainder > 0:
        for fast_client in fastest_clients:
            clients[fast_client]["training_examples_num"] += 1 #give the training example to the fastest clients
            total_allocations += 1

            if total_allocations == batch_size: #check again if number of allocated training examples = number of training examples
                break #stops loop (all training examples have been allocated)


    if not clients:
        return
    clients[0]["training_examples_start"] = 0 #first client starts at index 0

    for i in range(1, len(clients)): #skipping first client (start already assigned)
        client = clients[i]
        #starting position of client = previous starting position + previous number of training examples
        #(previous clients end position)
        client["training_examples_start"] = clients[i-1]["training_examples_num"] + clients[i-1]["training_examples_start"]

def handle_client(client): #function for handling and communicating with each client
    global recalibrate
    global state
    print("client connected:", client["address"])
    try:
        while client["connected"]:
            i = client["training_examples_start"] + batch_complete_addition
            j = client["training_examples_num"]
            if state == TRAINING and not client["finished"]:
                send_object(client["socket"], "TRAINING")
                send_object(client["socket"], get_model())
                selected_files = files[i:i+j] #all filenames within window
                selected_data = []
                selected_labels = []
                for filename in selected_files:
                    data = np.load(os.path.join("processed", filename))
                    selected_data.append(data) #np data from each file
                for file in selected_files: #giving label based on cat or dog status
                    if file[0] == "C":
                        selected_labels.append(1)
                    if file[0] == "D":
                        selected_labels.append(0)

                send_object(client["socket"], selected_data)
                send_object(client["socket"], selected_labels)

                print("Waiting for gradients from", client["address"])
                gradients = receive_object(client["socket"])

                if gradients is None:
                    print("client disconnected:", client["address"])
                    client["connected"] = False
                    client["socket"].close()
                    recalibrate = True
                    break

                print("Received gradients from", client["address"]) #below disconnect check

                #store summed gradient somewhere
                client["gradient"] = gradients
                client["finished"] = True
            time.sleep(0.01) #small delay to prevent crashing after every cycle

    except ConnectionError:
        print("client disconnected:", client["address"])
        client["connected"] = False
        client["socket"].close()
        recalibrate = True

def accept_clients(): #allows for continued acceptance of new clients in the background
    global recalibrate
    while True: #accepts all connections and creates a new socket for each
        client_socket, address = server.accept()
        while state == CALIBRATING:
            time.sleep(0.01)
        client = {
            "socket": client_socket,
            "address": address,
            "connected": True,
            "gradient": None,
            "training_examples_start": 0,
            "training_examples_num": 0,
            "finished": True, #starts true to allow recalibration when joining mid batch
        }
        clients.append(client)
        recalibrate = True

        #creates a sperate thread to run client_handle() in the background
        #allowes server to continue accepting new clients while still handling this client
        thread = threading.Thread(target=handle_client, args=(client,)) #takes in function and the functions arguments (in this case client tuple)
        #begin this thread
        thread.start()

def no_clients():
    global state
    global recalibrate

    print("no clients connected: saving model")

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

    state = WAITING
    recalibrate = False

def training_loop():
    global model
    global state
    global recalibrate
    global clients
    global start_time

    while True:
        if state == TRAINING:
            if clients and all(
                client["finished"] and client["gradient"] is not None
                for client in clients
            ):
                gradients = [client["gradient"] for client in clients]

                compute_gradients(gradients, start_time)

                for client in clients:
                    client["finished"] = False
                    client["gradient"] = None

                start_time = time.time()

        if recalibrate:
            #wait for the current training round to finish(for connected clients)
            #prevents multiple server threads from communicating with this socket at the same time e.g. handle client and calibrate workload (if another person joins)
            connected_clients = [client for client in clients if client["connected"]]
            if connected_clients and not all(client["finished"] for client in connected_clients):
                time.sleep(0.01)
                continue

            state = CALIBRATING
            #remove disconected clients
            clients = [client for client in clients if client["connected"]] #list comprehension (overwrites client lsit with cleints list without disconnected clients)

            if not clients:
                no_clients()
                continue

            print("calibrating workload")
            calibrate_workload(batch_size)

            for client in clients: #prevents any changes to model
                client["finished"] = False

            recalibrate = False
            start_time = time.time()
            state = TRAINING #resume training

#<==========server setup==========>

#create socket - AF_INET(means using ipv4 ip address) -SOCK_STREAM(means using TCP protocol)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 5000)) #0.0.0.0 -listen from all interfaces(ip addresses), 5000-listen on port 5000 -(could be any out of use port)

server.listen() #listen on port 5000

#start accepting clients
threading.Thread(
    target=accept_clients,
    daemon=True
).start()
# daemon=True makes this background thread close automatically when the main program exits.
# Needed here because accept_clients() runs forever waiting for new connections.
# Client threads don't need it because handle_client() ends naturally when the socket disconnects.

#start training loop
training_loop()