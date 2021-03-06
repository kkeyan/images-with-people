import os
import numpy as np
import cv2
import argparse
from PIL import Image
face_cascade = cv2.CascadeClassifier('/haarcascades/haarcascade_frontalface_alt2.xml')
parser = argparse.ArgumentParser()
parser.add_argument("-s", '--source', required=True)
parser.add_argument("-d", '--destination', required=True)
args = parser.parse_args()
path=args.source
dest=args.destination
def getImages(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    n = (len([os.path.join(path, f) for f in os.listdir(dest)]))
    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        extension = os.path.splitext(imagePath)[1]
        if (img is None):
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if (len(faces) > 0):
            n=n+1
            cv2.imwrite(dest+"/Img_"+str(n)+extension,img)
            cv2.waitKey(10)
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow("img", img)


getImages(path)
cv2.destroyAllWindows()
