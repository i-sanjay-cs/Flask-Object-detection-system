import cv2
import numpy as np

net = cv2.dnn.readNet('yolov4-custom.cfg', 'yolov4.weights')



with open("coco.names", "r") as f:
    classes = f.read().splitlines()
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))


class VideoLive(object):
    def __init__(self):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.cap=cv2.VideoCapture(0)
    def __del__(self):
        self.cap.release()

    def get_frame(self):
        _, img = self.cap.read()
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layeroutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in layeroutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
            # it will remove the duplicate detections in our detection
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 0, 0), 2)
        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
scale_factor = 1.3


class Face(object):
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
    def __del__(self):
        self.cap.release()
    def get_face(self):
        _, img = self.cap.read()
        faces = face_cascade.detectMultiScale(img, scale_factor, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'FACE', (x, y), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        _, jpeg = cv2.imencode('.jpeg', img)
        return jpeg.tobytes()




