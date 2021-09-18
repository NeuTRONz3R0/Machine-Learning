import cv2
import os
import sys

video_capture = cv2.VideoCapture(0)
#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(r"C:\\Users\\user\Downloads\\New folder\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2")
address = "http://192.168.1.104:4747/video"
video_capture.open(address)


i=0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('alien', frame)

    #saving image
    if cv2.waitKey(1) & 0xFF == ord('c'):
        directory = r'E:\\code stuff\\machine learning\\images'
        cv2.imwrite('saving...'+str(i)+'.jpg', frame)
        os.chdir(directory)
        list_of_saved_images = os.listdir(directory) #this is our data ! now lemme join again from phone 
        print('Successfully saved')
        i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()