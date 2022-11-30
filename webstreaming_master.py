from imutils.video import VideoStream
from flask import Response
from flask import Flask, request, redirect
from flask import render_template
import threading
import argparse
import time
import cv2
import argparse
import cv2
import time
from yoloface import face_analysis
from mtcnn.mtcnn import MTCNN

# New Additions
import os
import face_recognition
import numpy as np
from facenet_pytorch import InceptionResnetV1
import cv2
import torchvision.transforms as transforms
import torch

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
vs = VideoStream(src=0).start()

# WEBCAM_IP = 'http://192.168.43.1:8080/video'
#vs = VideoStream(src=WEBCAM_IP).start()

time.sleep(2.0)
dmodelchosen = None
rmodelchosen = None

def crop_photos(dmodelchosen):
    folder = './photos'
    savefolder = './cropped_photos'
    for filename in os.listdir(folder):
        indfolder = folder + "/" + filename
        indsavefolder = savefolder + "/" + filename
        for fname in os.listdir(indfolder):
            opencv_image = cv2.imread(os.path.join(indfolder, fname))
            if opencv_image is not None:
                if dmodelchosen == 'dmodel1':
                    tensor,box = yoloface(opencv_image)
                else:
                    tensor, box = mtcnn_detect(opencv_image)
                try:
                    for i in range(len(box)):
                        cropped = opencv_image[int(box[i][1]):int(
                            box[i][1])+int(box[i][3]), int(box[i][0]):int(box[i][0])+int(box[i][2])]
                        cropped = cv2.resize(cropped, (160, 160))
                        if not os.path.exists(indsavefolder):
                            os.makedirs(indsavefolder)
                        finalpath = indsavefolder + "/" + fname + str(i) + ".jpg"
                        cv2.imwrite(finalpath, cropped)
                except:
                    pass



def initialize_recognition():
    path = "./cropped_photos"
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    transform = transforms.ToTensor()

    known_encodings = []
    known_names = []

    for f in os.listdir(path):
        newpath = path + "/" + f
        for g in os.listdir(newpath):
            finalpath = newpath + "/" + g
            image = cv2.imread(finalpath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (160, 160))
            tensor = transform(image)

            face_encoding = resnet(tensor.unsqueeze(0))

            if(len(face_encoding) > 0):
                known_encodings.append(face_encoding.detach())
                known_names.append(f)

    return known_encodings, known_names

@app.route("/", methods=["GET", "POST"])
def home():
    if(request.method == 'POST'):
        global dmodelchosen, rmodelchosen
        dmodel = request.form.get('dmodel')
        rmodel = request.form.get('rmodel')
        dmodelchosen = dmodel
        rmodelchosen = rmodel
        return render_template("setup.html", dmodel=dmodel, rmodel=rmodel)
    else:
        return render_template("starter.html")

@app.route("/update")
def update():
    return render_template("update.html")

@app.route("/feed")
def index():
    t.start()
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        f = request.files["file"]
        saveloc = "./photos/" + request.form["name"]
        aadharid = request.form["aadhar"]

        if not os.path.exists(saveloc):
            os.makedirs(saveloc)
        f.save(os.path.join(saveloc, f.filename))

        return redirect("/")


def switch_coordinates(boxes_arr):
  finallist = []
  for box in boxes_arr:
    temp = [box[0],box[1],box[3],box[2]]
    finallist.append(temp)
  return finallist


def yoloface(frame):
    face = face_analysis()
    transform = transforms.ToTensor()
    tensor = []
    img, box, conf = face.face_detection(
           frame_arr=frame, frame_status=True, model='full')

    if box is not None:
        try:
            for i in range(len(box)):
                cropped = frame[int(box[i][1]):int(
                        box[i][1])+int(box[i][2]), int(box[i][0]):int(box[i][0])+int(box[i][3])]
                cropped = cv2.resize(cropped, (160, 160))
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                tensor.append(transform(cropped))
        except:
            pass
        box = switch_coordinates(box)
    return tensor, box

def mtcnn_detect(frame):
    mtcnn = MTCNN()
    boxes = []
    transform = transforms.ToTensor()
    detector = mtcnn.detect_faces(frame)
    tensor_arr = []
    for i in range(len(detector)):
        boxes.append(detector[i]['box'])
        box = detector[i]['box']
        cropped = frame[int(box[1]):int(
                        box[1])+int(box[3]), int(box[0]):int(box[0])+int(box[2])]
        cropped = cv2.resize(cropped, (160, 160))
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        tensor_arr.append(transform(cropped))
    return tensor_arr, boxes


def delete_old():
    folder = './cropped_photos'
    for filename in os.listdir(folder):
        indfolder = folder + "/" + filename
        for fname in os.listdir(indfolder):
            os.remove(os.path.join(indfolder, fname))
        os.rmdir(indfolder)


def make_rect(frame,box,name):
    x,y,h,w = box
    cv2.rectangle(frame, (int(x), int(y) + int(h) - 35),
                                    (int(x) + int(w), int(y) + int(h)), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (int(x) + 6, int(y) + int(h) - 6),
                                    font, 1.0, (255, 255, 255), 1)

    cv2.rectangle(frame, (int(x), int(y)), (int(
                            x)+int(w), int(y)+int(h)), (255, 0, 0), 2)


def detect_Persons(frameCount):
    global vs, outputFrame, lock, dmodelchosen, rmodelchosen
    delete_old()
    crop_photos(dmodelchosen)
    known_encodings, known_names = initialize_recognition()
    transform = transforms.ToTensor()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    while True:
        frame = vs.read()
        frame = cv2.resize(frame, (900, 600),
                           interpolation=cv2.INTER_AREA)

        if dmodelchosen == "dmode1":
            tensor, box = yoloface(frame)
        else:
            tensor, box = mtcnn_detect(frame)
        
        try:
            if tensor is not None:
                for i in range(len(box)):
                        face_encoding = resnet(tensor[i].unsqueeze(0)).detach()

                        dist_list = []
                        for idx, emb_db in enumerate(known_encodings):
                            dist = torch.dist(face_encoding, emb_db).item()
                            dist_list.append(dist)

                        min_dist = min(dist_list) 
                        min_dist_idx = dist_list.index(
                            min_dist) 
                        name = known_names[min_dist_idx]

                        if min_dist > 0.9:
                            name = "Unknown"
            
                        make_rect(frame,box[i],name)
                        
        except:
            pass

        outputFrame = frame.copy()


def generate():
    global outputFrame, lock
    while True:
        if outputFrame is None:
            continue
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    DETECTION_FRAME_THICKNESS = 1
    OBJECTS_ON_FRAME_COUNTER_FONT = cv2.FONT_HERSHEY_SIMPLEX
    OBJECTS_ON_FRAME_COUNTER_FONT_SIZE = 1
    LINE_COLOR = (0, 0, 255)
    LINE_THICKNESS = 3
    LINE_COUNTER_FONT = cv2.FONT_HERSHEY_DUPLEX
    LINE_COUNTER_FONT_SIZE = 2.0
    LINE_COUNTER_POSITION = (20, 45)
    targeted_classes = ['person']
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    t = threading.Thread(target=detect_Persons, args=(
        args["frame_count"],))
    t.daemon = True
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
vs.stop()
