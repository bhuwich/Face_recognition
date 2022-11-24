import cv2
import time
from threading import Thread
import os
import numpy as np
import face_recognition

class Stream:
    def __init__(self, protocol):
        self.protocol = protocol
        self.cap = cv2.VideoCapture(self.protocol)
        self.frame = None
        self.run_flag = False
    
    def readStream(self):
        while self.run_flag:
            ret,frame = self.cap.read()
            if ret:
                self.frame = frame
    
    def read(self):
        return self.frame

    def start(self):
        self.run_flag = True
        Thread(target=self.readStream, args=()).start()

    def stop(self):
        self.run_flag = False
        time.sleep(1)

class FaceRecognition:
    def __init__(self, dataset_dir):
        self.names = [name for name in os.listdir(dataset_dir) if name != '.ipynb_checkpoints']
        data = {}
        for name in self.names:
            data[name] = []
            filenames = [x for x in os.listdir(f"{dataset_dir}/{name}") if x != '.ipynb_checkpoints']
            for i in range(len(filenames)):
                filename = f"{dataset_dir}/{name}/{filenames[i]}"
                print(f"encoding ... {filename}")
                image = cv2.imread(filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # แปลง image เป็น RGB
                locations = face_recognition.face_locations(image)
                top, right, bottom, left = locations[0]
                face = image[top:bottom, left:right, :]
                face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, face.shape[1]-1, face.shape[0]-1,0)])[0]
                data[name].append(face_encoding)
            data[name] = np.mean(data[name],axis=0)
        self.data = data
    
    def detectFace(self,image):
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(image)
        if len(locations) > 0:
            return locations[0]
        else:
            return []

    def predictFace(self,face):
        face = face.copy()
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(face, known_face_locations = [(0, face.shape[1]-1, face.shape[0]-1,0)])[0]
        names = list(self.data.keys())
        encodings = [self.data[name] for name in names]
        distances = face_recognition.face_distance(encodings, face_encoding)
        index_min = np.argmin(distances)
        return names[index_min]

if __name__ == '__main__':
    face_recog = FaceRecognition("data")
    cap = Stream(0)
    cap.start()
    
    while True:
        frame = cap.read()
        if frame is not None:
            locations = face_recog.detectFace(frame)
            if len(locations) > 0:
                top, right, bottom, left = locations
                face = frame[top:bottom, left:right, :]
                face_recog.predictFace(face)
                name = face_recog.predictFace(face)
                cv2.rectangle(frame, (left,top),(right,bottom),(0,255,0),2)
                cv2.putText(frame,name, (left,top),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.stop()
    cv2.destroyAllWindows()