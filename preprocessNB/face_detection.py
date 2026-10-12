import cv2
face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
img = cv2.imread(r"C:\Users\susha\deepfake detector\data\image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face.detectMultiScale(gray,1.3,5,minSize=(30, 30))
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()