import numpy as np
import cv2
import imutils
import os
import argparse
from imutils import paths
from imutils import contours
from imutils.perspective import four_point_transform

def initArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i" , "--dir" , required=True , help="Image Directory")
    return vars(parser.parse_args())

def out(image):
    cv2.imshow("Image" , image)
    cv2.waitKey(0)

if __name__ == "__main__":
    args = initArgs()
    images = list(paths.list_images(args["dir"]))
    print(images)

    answer_key = {
        0:1,
        1:4,
        2:2,
        3:4,
        4:0
    }
    for zzz in images:
        image = cv2.imread(zzz)
        gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray , (5,5) , 0)
        edged = cv2.Canny(blurred , 100 , 200)

        # out(image)
        # out(gray)
        # out(blurred)
        # out(edged)

        cnts = cv2.findContours(edged.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        doccnt=None

        if len(cnts) > 0:
            cnts = sorted(cnts , key=cv2.contourArea , reverse=True)

            for c in cnts:
                peri = cv2.arcLength(c,True)
                approx = cv2.approxPolyDP(c , 0.02*peri , True)
                # print(approx)

                if len(approx) == 4:
                    doccnt = approx
                    break
        paper = four_point_transform(image , doccnt.reshape(4,2))
        warped = four_point_transform(gray , doccnt.reshape(4,2))

        # out(paper)
        # out(warped)

        thresh = cv2.threshold(warped , 0 , 255 ,  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # out(thresh)

        cnts = cv2.findContours(thresh.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []

        for c in cnts:
            (x,y,w,h) = cv2.boundingRect(c)
            ar = w / h

            if w>=20 and h>=20 and ar >= 0.9 and ar <= 1.1:
                questionCnts.append(c)
        
        
        questionCnts = contours.sort_contours(questionCnts , method="top-to-bottom")[0]
        correct = 0

        for (q,i) in enumerate(np.arange(0,len(questionCnts),5)):
            cnts = contours.sort_contours(questionCnts[i:i+5])[0]
            bubbled = None

            for (j,c) in enumerate(cnts):
                mask = np.zeros(thresh.shape , dtype="uint8")
                cv2.drawContours(mask , [c] , -1 , 255 , -1)

                mask = cv2.bitwise_and(thresh,thresh,mask=mask)
                total=cv2.countNonZero(mask)

                if bubbled is None or total > bubbled[0]:
                    bubbled = (total,j)
                
            color = (0,0,255)
            k = answer_key[q]
            if k == bubbled[1]:
                color = (0,255,0)
                correct+=1
            cv2.drawContours(paper , [cnts[k]] , -1 , color , 3)

        score = (correct/5) * 100
        print("Score : {:.2f}%".format(score))
        cv2.putText(paper , "{:.2f}%".format(score) , (10,30) , cv2.FONT_HERSHEY_SIMPLEX , 0.9 , (0,0,255) , 3)
        # out(image)
        out(paper)