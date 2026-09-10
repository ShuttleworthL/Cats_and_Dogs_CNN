from PIL import Image #for image processing
import numpy as np #for vector and matrix math
import os

#preprocess all images - conform to set resolution and aspect ratio

input_size = 240 #240x240 pixels (240x240x3 - for rgb channels)
folders = ["Cat", "Dog"] #two source folders

for i in range(12500): #all images in folder
    for folder in folders: #for both cats and dogs folders
        #produces cat,dog,cat,dog alterantion
        img = Image.open("Petimages/" + folder + "/" + str(i) + ".jpg") #individual image file - str(i) means the number i in string format e.g. "5"
        img = img.convert("L") #force graysclae format

        ratio = min( #returns smallest value
            input_size / img.width,
            input_size / img.height
        ) #creates scale factor while maintaining resolution (minimum change in size to both axis, to get all axis within or at 240px)
        width = int(img.width * ratio) #new width and height (rounded down to nearest integer)
        height = int(img.height * ratio)

        img = img.resize(
            (width, height),
            Image.Resampling.LANCZOS #resizing algorithm
        ) #resizes image to within 240x240 without stetching (maintainig aspect ratio)

        canvas = Image.new( #creates black 240x240 rgb image
            "L",
            (input_size, input_size),
            (0)
        )
        #calculate where to center image over black canvas (creates letter boxing around image to create perfect 240x240 shape)
        x = (input_size - width) // 2
        y = (input_size - height) // 2
        canvas.paste(img, (x, y)) #pastes and centers image over canvas

        image_array = np.array(canvas) #converts final image into 240x240x3 array (x3 is for rgb channels)
        np.save( #saves processed image arrays into Processed folder
            "Processed/" + folder + "_" + str(i) + ".npy",
            image_array
        )
        if i % 1000 == 0: #modulus - remainder of a division (e.g. i = 2000, 2000/1000 = remainder 0) -multiples of 1000
            print(folder, i)