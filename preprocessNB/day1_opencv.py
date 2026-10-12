import cv2

image = cv2.imread(r"C:\Users\susha\deepfake detector\data\rahulphoto1.jpg")

# print(image)
print(image.shape)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Original shape:", image.shape)
print("Gray shape:", gray.shape)


save=cv2.imwrite(r"C:\Users\susha\deepfake detector\data\rahulphoto1_gray.jpg", gray)
# cv2.imshow("Grayscale", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("My Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()