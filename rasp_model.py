#WARNING: The following code is not valid on Windows.
#Export this code to a raspberry pi to run it there.
#This code is for running the model on a raspberry pi with a camera module.
from picamera2 import Picamera2
import cv2
import onnxruntime as ort
import numpy as np
import uuid

picam = Picamera2()
picam.preview_configuration.main.size = (224,224)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()
picam.configure("preview")
picam.start()


session = ort.InferenceSession("best.onnx")
labels = {0:"battery",1:"biological",2:"brown-glass",3:"cardboard",4:"clothes",5:"green-glass",6:"metal",7:"paper",8:"plastic",9:"shoes",10:"trash",11:"white-glass"}
while True:
    frame = picam.capture_array()
    cv2.imshow("trash classifier", frame)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break

    if key==ord('c'):
        i = str(uuid.uuid4())[:8]
        rgb_frame = cv2.cvtColor(frame.copy(),cv2.COLOR_RGB2BGR)
        scaled_frame = np.array(frame).astype(np.float32)/255.0
        transposed_frame = np.transpose(scaled_frame,(2,0,1))
        frames = np.expand_dims(transposed_frame,axis=0)
        outputs = session.run(None,{"images":frames})
        label = labels[np.argmax(outputs[0])]
        print("predicted class: ",label)
        cv2.putText(frame,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imwrite(f"captured_image{i}.jpg",frame)
cv2.destroyAllWindows()
picam.close()