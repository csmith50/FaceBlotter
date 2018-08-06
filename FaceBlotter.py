import numpy as np
import cv2  
from tkinter import * #for our gui
import dlib
from helperFunctions import *

def saveFace():
    global root
    root.withdraw() #hide our root to clear screen space
    new = Toplevel() #open a new window seperate from root
    new.title("Face Blotter")
    new.resizable(0, 0)
    new.geometry("300x150")

    confirm = Label(new, text = "Press 's' once desired face is in frame.")
    confirm.place(x = 150, y = 40, anchor = "center")

    confirmButton = Button(new, text = "OK", command = new.destroy)
    confirmButton.place(x = 150, y = 90, anchor = "center")

    root.wait_window(new) #have the program wait until the user has clicked ok
    
    cap = cv2.VideoCapture(0)
    properExit = False #false if the user hits q to exit and true if the user hits s
    while True:
        ret, frame = cap.read()
        
        #display the feed
        cv2.imshow("Landmark detection", frame)

        #check if the user has entered a 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #check for a s key from the user
        if cv2.waitKey(1) & 0xFF == ord('s'):
            properExit = True
            break
            
    
    cap.release()
    cv2.destroyAllWindows()
    #save at this point
    if properExit:
        cv2.imwrite("savedFaces.jpg", frame)

        notif = Toplevel()
        notif.title("Face Blotter")
        notif.resizable(0, 0)
        notif.geometry("150x150")

        notifText = Label(notif, text = "Image saved successfully.")
        notifText.place(x = 75, y = 40, anchor = "center")
        confirmButton = Button(notif, text = "OK", command = notif.destroy)
        confirmButton.place(x = 75, y = 100, anchor = "center")
        root.wait_window(notif)


    root.deiconify() #continue the program loop

def blotFace():
    global root
    root.withdraw() #hide our root to clear screen space

    detector = dlib.get_frontal_face_detector() #detect a face from the webcam
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #find the landmarks on the face
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat") #convert face into 128D vector to compare other faces to
    
    #open savedFaces.jpg
    savedFaces = cv2.imread("savedFaces.jpg")

    #perform facial landmark and recognition calculations
    gray = cv2.cvtColor(savedFaces, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1) #isolate the faces from the image
    shapes = []
    descriptors = []
    for (i, rect) in enumerate(rects): #loop through each face detected
        shape = predictor(gray, rect)
        shapes.append(shape)
        descriptors.append(facerec.compute_face_descriptor(savedFaces, shape))

    #open the webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect faces and perform facial recognition calculations
        currentRects = detector(gray, 1)
        currentShapes = []
        currentDesecriptors = []
        for (i, rect) in enumerate(currentRects):
            shape = predictor(gray, rect)
            currentShapes.append(shape)
            currentDesecriptors.append(facerec.compute_face_descriptor(frame, shape))

        for (i, desc) in enumerate(currentDesecriptors):
            sameFace = False
            for (j, descSaved) in enumerate(descriptors): #search through our known faces and compare 
                sameFace = compareFaces(descSaved, desc, 0.6) 
                #print("Result from comparision: " + sameFace)
                if sameFace != True: #draw a black rectangle around face we don't recognize
                    (x, y, w, h) = rect_to_bb(currentRects[i])
                    cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 0, 0), -1)
                    #coords = shape_to_np(currentShapes[i])
                    #radius = coords[14][0] - coords[2][0]
                    #cv2.circle(frame, (coords[33][0],coords[33][0]), radius, (0, 0, 0), -1)
                    

        cv2.imshow("Face Blotter", frame)

        #check if the user has entered a 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.deiconify() #continue the program loop


#globals
root = Tk()
root.title("Face Blotter")
root.resizable(0, 0)
root.geometry("200x200")
frame = Frame(root)
frame.pack_propagate(0)
frame.pack(fill = BOTH, expand = 1)

instructionText = Label(frame, text = "What would you like to do?")
instructionText.place(x = 90, y = 60, anchor = "center")

saveButton = Button(frame, text = "Save a face", command = saveFace)
saveButton.place(x = 90, y = 110, anchor = "center")

blotButton = Button(frame, text = "Look for saved faces", command = blotFace)
blotButton.place(x = 90, y = 140, anchor = "center")

root.mainloop()
