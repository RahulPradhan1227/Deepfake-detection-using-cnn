import cv2
img=cv2.imread(r"C:\Users\susha\deepfake detector\data\rahulphoto1.jpg")

croped =img[100:200,100:200]
cv2.imwrite(r"C:\Users\susha\deepfake detector\data\rahulphoto1_croped.jpg",croped)
cv2.imshow("rahul's croped img",croped)
cv2.waitKey(0)
cv2.destroyAllWindows()  