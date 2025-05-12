import cv2
import numpy as np
from picamera2 import Picamera2
import tensorflow as tf

model = tf.keras.models.load_model("/home/jonliew/Desktop/symbol_recognition_model.h5")

image_size = (224, 224)


class_names = [line.strip().split(" ", 1)[1] for line in open("/home/jonliew/Desktop/labels.txt", "r").readlines()]
  

picam2 = Picamera2()
configure = picam2.create_preview_configuration(main={"size": (640,480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()

    img = cv2.resize(frame, image_size)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  

    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    label = class_names[class_index]
        
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    print("{}".format(label))
    cv2.imshow("Symbol Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #without this line, the camera wont open because no delay.. THis is technically a delay
        break
cap.release()
cv2.destroyAllWindows()