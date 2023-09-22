import cv2

cap = cv2.VideoCapture(-1)
num = 0

while cap.isOpened():
    success, img = cap.read()
    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print('Image saved!')
        num += 1

    cv2.imshow('Img', img)

cap.release()
cv2.destroyAllWindows()
