import cv2
import os
import numpy as np
dataPath = "C:/Users/PC/Desktop/Probando_face_recognition/Image"#Cambia a la ruta donde hayas almacenado Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)
labels = []
facesData = []
label = 0
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        #image = cv2.imread(personPath+'/'+fileName,0)
        #cv2.imshow('image',image)
        #cv2.waitKey(10)
    label = label + 1
'''
#Generar base de datos
import cv2
import os
import imutils
personName = 'Javier'
dataPath = 'C:/Users/PC/Desktop/Probando_face_recognition/Image'#Cambia a la ruta donde hayas almacenado Data
personPath = dataPath + '/' + personName
if not os.path.exists(personPath):
    print('Carpeta creada: ',personPath)
    os.makedirs(personPath)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('Video.mp4')
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0
while True:
    
    ret, frame = cap.read()
    if ret == False: break
    frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    faces = faceClassif.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
        count = count + 1
    cv2.imshow('frame',frame)
    k =  cv2.waitKey(1)
    if k == 27 or count >= 300:
        break
cap.release()
cv2.destroyAllWindows()
'''
face_recognizer = cv2.face.EigenFaceRecognizer_create()
print("Entrenado...")
face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloEigenFace.xml')
print("Modelo almacenado")