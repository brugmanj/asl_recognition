import cv2
from tensorflow.keras.models import load_model
import numpy as np

cap = cv2.VideoCapture(0)
i = 1
# Loading CNN
model = load_model('./code/model1.h5')

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    
    # Formatting image properly
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    img = img /255

    # Cropping the image to the proper size
    img = img[230:830, 660:1260]
    resized = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)

    if i == 1:
        print(resized.shape)
        i += 1

    # Resizing the image, preparing for input to CNN
    test_img = img.reshape(1, 200, 200, 1)

    # Predicting with CNN
    pred = model.predict_classes(test_img)
    letter = chr(pred+65)

    # Drawing gides on output image
    frame = cv2.rectangle(frame, 
                        (660, 230), 
                        (1260, 830), 
                        (0, 255, 0), 
                        2)

    font = cv2.FONT_HERSHEY_SIMPLEX 
  
    # Use putText() method for 
    # inserting text on video 
    cv2.putText(frame,  
                letter,  
                (200, 100),  
                font, 4,  
                (0, 255, 0),  
                4,  
                cv2.LINE_4) 
    frame = cv2.resize(frame, (960, 540), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', resized)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()