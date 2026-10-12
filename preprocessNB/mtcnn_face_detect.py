import cv2
from mtcnn import MTCNN
# Load image
img = cv2.imread(r"C:\Users\susha\deepfake detector\data\image.png")
# Convert BGR to RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Create MTCNN detector
detector = MTCNN()
# Detect faces
faces = detector.detect_faces(rgb_img)
# Draw rectangle around each face
for face in faces:
    x, y, w, h = face["box"]
    cv2.rectangle(img,(x, y),(x + w, y + h),(255, 0, 0),2)
# Display result
cv2.imshow("MTCNN Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()