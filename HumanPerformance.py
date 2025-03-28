from tkinter import *
import tkinter
import numpy as np
import imutils
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
from keras.preprocessing import image
from tkinter import filedialog
from tkinter.filedialog import askopenfilename


main = tkinter.Tk()
main.title("Machine Learning for Efficient Assessment and Prediction of Human Performance in Collaborative Learning Environments")
main.geometry("800x500")

global cnn_model
global video

def loadModel():#load cnn model
    global cnn_model
    img_width, img_height = 150, 150
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Convolution2D(32, 3, 3, activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(output_dim = 128, activation = 'relu'))
    cnn_model.add(Dense(output_dim = 10, activation = 'softmax'))
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    cnn_model.load_weights('model/cnn_model.h5')
    pathlabel.config(text="          CNN Model Generated Successfully")
    
    
def upload():#upload video as input
    global video
    filename = filedialog.askopenfilename(initialdir="Video")
    pathlabel.config(text="          Video loaded")
    video = cv.VideoCapture(filename)
    
#track or assess human performance
def humanPerformance():
    global cnn_model
    while(True):
        ret, frame = video.read()
        print(ret)
        if ret == True:
            cv.imwrite("test.jpg",frame)
            imagetest = image.load_img("test.jpg", target_size = (150,150))
            imagetest = image.img_to_array(imagetest)
            imagetest = np.expand_dims(imagetest, axis = 0)
            predict = cnn_model.predict_classes(imagetest)
            print(predict)
            msg = "";
            if str(predict[0]) == '0':
                msg = 'Safe Driving'
            if str(predict[0]) == '1':
                msg = 'Using/Talking Phone'                        
            if str(predict[0]) == '2':
                msg = 'Talking On phone'                        
            if str(predict[0]) == '3':
                msg = 'Using/Talking Phone'                        
            if str(predict[0]) == '4':
                msg = 'Using/Talking Phone'                        
            if str(predict[0]) == '5':
                msg = 'Drinking/Radio Operating'                        
            if str(predict[0]) == '6':
                msg = 'Drinking/Radio Operating'                        
            if str(predict[0]) == '7':
                msg = 'Reaching Behind'                        
            if str(predict[0]) == '8':
                msg = 'Hair & Makeup'                        
            if str(predict[0]) == '9':
                msg = 'Talking To Passenger'                        
            text_label = "{}: {:4f}".format(msg, 80)
            cv.putText(frame, text_label, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.imshow('Frame', frame)
            if cv.waitKey(2500) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()


def exit():
    global main
    main.destroy()
  

font = ('times', 16, 'bold')
title = Label(main, text='                                                                                                     Real-Time Feedback Mechanism Using Machine Learning',anchor=W, justify=LEFT)
title.config(bg='Navy', fg='white')  
title.config(font=font)           
title.config(height=3, width=150)       
title.place(x=0,y=5)

# Define bottom margin and spacing between elements
bottom_margin = 50  # Space from bottom of window 
button_spacing = 50  # Vertical space between buttons

# Calculate y-positions from bottom up
exit_y = main.winfo_height() - bottom_margin
predict_y = exit_y - button_spacing
upload_y = predict_y - button_spacing
path_y = upload_y - button_spacing
load_y = path_y - button_spacing

font1 = ('times', 13, 'bold')
loadButton = Button(main, text="Generate & Load CNN Model", command=loadModel)
loadButton.place(x=350, y=200)
loadButton.config(font=font1)  

uploadButton = Button(main, text="Upload Input Video", command=upload)
uploadButton.place(x=350, y=300)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkSalmon', fg='firebrick4')  
pathlabel.config(font=font1)           
pathlabel.place(x=650, y=305)

predictButton = Button(main, text="Predict Human Performance", command=humanPerformance)
predictButton.place(x=350, y=400)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=350, y=500)
exitButton.config(font=font1)

main.config(bg='Cyan')
main.mainloop()
