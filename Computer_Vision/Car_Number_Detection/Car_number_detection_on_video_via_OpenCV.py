# Car number detection on video
import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as plt

videofile_path = 'videos/Car_OpenCV.mp4'

video = cv2.VideoCapture(videofile_path)

while True:
    success, img = video.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filter_noise = cv2.bilateralFilter(gray, 3, 11, 11)

    edges = cv2.Canny(filter_noise, 30, 200)  #finding all corners in the img

    #finding all contours
    cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)

    #sorting finded contours and selecting only squared contours
    cont = sorted(cont, key = cv2.contourArea, reverse = True)
    pos = None

    for curve in cont:
        approx = cv2.approxPolyDP(curve, 6, True)
        if len(approx == 4):
            pos = approx
            break

    #create mask
    mask = np.zeros(gray.shape, np.uint8)
    new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
    bitwise_img = cv2.bitwise_and(img, img, mask = mask)

    #crop only part image
    x, y = np.where(mask == 255)

    x1, y1 = np.min(x), np.min(y)
    x2, y2 = np.max(x), np.max(y)

    crop = gray[x1:x2, y1:y2]  #slicing

    #read info from car number
    text = easyocr.Reader(['en'])
    text = text.readtext(crop)
##    print(text)

    #show info on video
    if len(text) > 1:
        res_text = text[0][-2] + ' ' + text[1][-2]
    ##    print(res_text)
        final_image = cv2.putText(img, res_text, (y2, x2 + 40), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 1)
        final_image = cv2.rectangle(img, (y1, x1), (y2, x2), (0, 255, 0), 1)
    ##    print(type(final_image))
    else:
        final_image = img  #if is not car number detected on image -> output just the image

    #output
    cv2.imshow('Result video with detection car number', final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
##    cv2.waitKey(0)
##    break
