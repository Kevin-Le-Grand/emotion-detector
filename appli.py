import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
from keras.models import load_model


emotion_model = load_model("emotions_aug.h5")
emotion_labels = ['Angry', 'Disgust','Fear','Happy', 'Sad', 'Surprise', 'Neutral']
emotion_images = {
    'Happy': cv2.imread('emojis/happy.png'),
    'Sad': cv2.imread('emojis/sad.png'),
    'Angry': cv2.imread('emojis/angry.png'),
    'Disgust': cv2.imread('emojis/disgust.png'),
    'Fear': cv2.imread('emojis/fear.png'),
    'Surprise': cv2.imread('emojis/surprise.png'),
    'Neutral': cv2.imread('emojis/neutral.png')    }

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
aff=1
while True:
    # Capture une frame de la webcam
    ret, frame = cap.read()
    
    grey_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(grey_image)
    
    
    #Boucle à travers les visages détectésq
    for (x, y, w, h) in faces:
        # Extraire le visage de la frame
        face_cropped = grey_image[y:y+h, x:x+w]
        print("face",face_cropped.shape)
        # Redimensionner le visage pour correspondre à l'entrée du modèle
        face_resized = cv2.resize(face_cropped,(48,48),interpolation = cv2.INTER_AREA)
        face_resized = face_resized.reshape(-1,48,48,1)/255.0  
        # Prédire l'émotion du visage
        emotion = emotion_labels[np.argmax(emotion_model.predict(face_resized))]
        # Trouver le smiley correspondant
        emotion_image = emotion_images[emotion]

        # Affichage du mode d'emploi
        cv2.putText(frame, "Choix du mode : 'a' ,'z' ou 'e'", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Pressez 'q' pour quitter.", 
                    (10, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Selection du mode
        if aff==1:
            # Redimensionner l'image pour s'adapter à la taille du visage
            emotion_image = cv2.resize(emotion_image, (w, h),
                            interpolation=cv2.INTER_CUBIC)
            # Afficher l'imade de l'emotion prédite
            frame[y:y+h, x:x+w] = emotion_image
        elif aff==2:
            # Afficher l'émotion prédite sur la frame en texte
            cv2.putText(frame, emotion, (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            emotion_image = cv2.resize(emotion_image, (w, h),
                            interpolation=cv2.INTER_CUBIC)
            frame[y:y+h, x:x+w] = emotion_image
            cv2.putText(frame, emotion, (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
                

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('a'):
        aff=1
    if key == ord('z'):
        aff=2
    if key == ord('e'):
        aff=3
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
       
cap.release()
cv2.destroyAllWindows()
