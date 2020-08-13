import cv2
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)

# Loading CNN
model = load_model('./code/model1.h5')

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    
    # Formatting image properly
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = img.reshape(1, 200, 200, 1)
    img = img /255

    pred = model.predict_classes(img)
    letter = chr(pred+65)

    font = cv2.FONT_HERSHEY_SIMPLEX 
  
    # Use putText() method for 
    # inserting text on video 
    cv2.putText(frame,  
                letter,  
                (200, 100),  
                font, 4,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
    frame = cv2.resize(frame, (200, 200), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()