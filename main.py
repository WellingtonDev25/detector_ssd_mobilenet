import cv2
import numpy as np
import cvzone

# videoPath = 'example1.mp4'
videoPath = 0
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
modelPath = 'frozen_inference_graph.pb'
classesPath='coco.names'

net = cv2.dnn_DetectionModel(modelPath,configPath)

net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


with open(classesPath,'r') as f:
    classesList = f.read().splitlines()

video = cv2.VideoCapture(videoPath)

while True:
    check,img = video.read()
    img = cv2.resize(img,(1270,720))

    labels, confs, bboxs = net.detect(img,confThreshold=0.5)

    bboxs = list(bboxs)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))

    bboxsIdx = cv2.dnn.NMSBoxes(bboxs,confs,score_threshold=0.5, nms_threshold=0.3)

    if len(bboxsIdx) !=0:
        for x in range(0,len(bboxsIdx)):
            bbox = bboxs[np.squeeze(bboxsIdx[x])]
            conf = confs[np.squeeze(bboxsIdx[x])]
            labelsID = np.squeeze(labels[np.squeeze(bboxsIdx[x])])-1
            label = classesList[labelsID]

            print(bbox,labelsID,conf)
            x,y,w,h = bbox

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            cvzone.putTextRect(img,f'{label} {round(conf,2)}',(x,y-10),colorR=(255,0,0),scale=1,thickness=2)

    cv2.imshow('Imagem',img)

    if cv2.waitKey(1)==27:
        break
