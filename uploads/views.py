from django.shortcuts import render,redirect
from .form import ImageForm
from .models import Image
import easyocr
from PIL import Image
import cv2
# from matplotlib import pyplot as plt
import numpy
import io
import base64
# Create your views here.
def index(request):
    console.log("B44 post")
    if request.method == 'POST':
        console.log("B4 post")
        form=ImageForm(request.POST, request.FILES)
        console.log("posted")
        if form.is_valid():
            console.log("validate")
            instance = form.save()
            IMAGE_PATH = instance.image.path
            #print(IMAGE_PATH)
            reader = easyocr.Reader(['en'])
            result = reader.readtext(IMAGE_PATH)

            font = cv2.FONT_HERSHEY_SIMPLEX

            images = cv2.imread(IMAGE_PATH)
            for detection in result:
                top_left = tuple(detection[0][0])
                bottom_right = tuple(detection[0][2])
                text = detection[1]
                images = cv2.rectangle(images, top_left, bottom_right, (0, 255, 0), 3)
                images = cv2.putText(images, text, top_left, font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
            #print(images)

            config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
            frozen_model = 'frozen_inference_graph.pb'
            model = cv2.dnn_DetectionModel(frozen_model,config_file)

            model.setInputSize(320,320)
            model.setInputScale(1.0/127.5)
            model.setInputMean((127.5,127.5,127.5))
            model.setInputSwapRB(True)

            classLabels = []
            file_name = 'Labels.txt'
            with open(file_name,'rt') as fpt:
                classLabels = fpt.read().rstrip('\n').split('\n')
            #print(classLabels)

            ClassIndex, confidece, bbox = model.detect(images,confThreshold=0.5)
            font_scale = 3
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
                cv2.rectangle(images,boxes,(255,0,0),2)
                cv2.putText(images,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color =(0,255,0),thickness=3)

            image = Image.fromarray(images, 'RGB')

            data = io.BytesIO()
            image.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
            out = encoded_img_data.decode('utf-8')


            return render(request,"index.html",{"image": out})
    else:
        form = ImageForm()
    return render(request,'index.html',{"form":form})

