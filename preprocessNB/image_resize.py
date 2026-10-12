import cv2
img=cv2.imread(r"C:\Users\susha\deepfake detector\data\rahulphoto1.jpg")

resized =cv2.resize(img,(300,300))
cv2.imwrite(r"C:\Users\susha\deepfake detector\data\rahulphoto1_resized.jpg",resized)
cv2.imshow("rahul's resized img",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()