import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def hacer_algo():
    print(" aqui va la accion para despertar al conductor ")
    # Aqui debe ir el codigo de los arduino para ejecutar la acción, revisar si es mejor un arduino o un raspberry
    #verificar las interrupciones que se pueden utilizar en el arduino, se requiere wifi, y alguna herramienta sonora

def drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter):
     aux_image = np.zeros(frame.shape, np.uint8)
     contours1 = np.array([coordinates_left_eye])
     contours2 = np.array([coordinates_right_eye])
     cv2.fillPoly(aux_image, pts=[contours1], color=(255, 0, 0))
     cv2.fillPoly(aux_image, pts=[contours2], color=(255, 0, 0))
     output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)

     cv2.rectangle(output, (0, 0), (200, 50), (0, 0, 0), -1)
     cv2.rectangle(output, (202, 0), (265, 50), (0, 0, 0),2)
     cv2.putText(output, "Total Parpadeos:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
     cv2.putText(output, "{}".format(blink_counter), (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
     
     return output
     
def eye_aspect_ratio(coordinates):
     d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
     d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
     d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))

     return (d_A + d_B) / (2 * d_C)
        
#esta linea es para linux windows donde requiere el segundo parametro para interactuar con la cámara
#cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#esta línea se uusa en mac, no se requiere el segundo parametro para el uso de la cámara
cap = cv2.VideoCapture(1)

mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.26
NUM_FRAMES = 2
aux_counter = 0
blink_counter = 0
line1 = []
pts_ear = deque(maxlen=64)
i = 0
aux_dormir = 0
with mp_face_mesh.FaceMesh(
     static_image_mode=False,
     max_num_faces=1) as face_mesh:

     while True:
          ret, frame = cap.read()
          if ret == False:
               break
          frame = cv2.flip(frame, 1)
          height, width, _ = frame.shape
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = face_mesh.process(frame_rgb)
          
          coordinates_left_eye = []
          coordinates_right_eye = []
           
          if results.multi_face_landmarks is not None:
               for face_landmarks in results.multi_face_landmarks:
                    for index in index_left_eye:
                         x = int(face_landmarks.landmark[index].x * width)
                         y = int(face_landmarks.landmark[index].y * height)
                         coordinates_left_eye.append([x, y])
                         cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                         cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                    for index in index_right_eye:
                         x = int(face_landmarks.landmark[index].x * width)
                         y = int(face_landmarks.landmark[index].y * height)
                         coordinates_right_eye.append([x, y])
                         cv2.circle(frame, (x, y), 2, (128, 0, 250), 1)
                         cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
                         
               ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
               ear_right_eye = eye_aspect_ratio(coordinates_right_eye)
               ear = (ear_left_eye + ear_right_eye)/2

               # Ojos cerrados
               if ear < EAR_THRESH:
                    aux_counter += 1

               else:
                    if aux_counter >= NUM_FRAMES:
                         aux_counter = 0
                         blink_counter += 1 

               frame = drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter)
               pts_ear.append(ear)

               i +=1
               if  ear < 0.20:
                    print ("XXXXXX OJOS CERRADOS XXXXXXX")
                    aux_dormir +=1
               if ear > 0.30:
                    print ("XXXXXXXX OJOS ABIERTOS XXXXXX")
                    aux_dormir = 0
               if aux_dormir > 10:
                    cv2.putText(frame, "CTM, TE QUEDASTE DORMIDO!!", (150, 300),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
                    hacer_algo()

          cv2.imshow("Medidor de Dormir", frame)
          k = cv2.waitKey(1) & 0xFF
          if k == 27:
               break
cap.release()
cv2.destroyAllWindows()
