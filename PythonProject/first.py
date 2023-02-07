import cv2

face_cascade = cv2.CascadeClassifier("classifier/haarcascade_frontalface_default.xml")

# initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # capture a frame
    ret, frame = cap.read()
    
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # show the frame
    cv2.imshow("Face Detection", frame)
    
    # check if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# release the video capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
