import cv2 as cv
import numpy as np

def tsd(frame):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    cv.imshow("hsv",hsv)
    blur = cv.blur(hsv,(5,5))
    cv.imshow("blur", blur)
    thresh=cv.inRange(blur,(89,124,73),(255,255,255))
    # cv.imshow("thresh", thresh)
    thresh=cv.erode(thresh,None,iterations=2)
    thresh= cv.dilate(thresh,None,iterations=4)
    cv.imshow("thresh", thresh)
    countours = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    countours = countours[1]
    height,weight=thresh.shape
    roiImg = []
    for cnt in countours:
        cnt = sorted(countours,key=cv.contourArea,reverse=True)[0]
        rect=cv.minAreaRect(cnt)
        box = np.int0(cv.boxPoints(rect))
        cv.drawContours(frame,[box],-1,(0,255,0),3)
        cv.imshow("frame",frame)
        (x,y,w,h)=cv.boundingRect(cnt)
        min_x,min_y=weight,height
        max_x=max_y=0
        min_x,max_x = min(x,min_x),max(x+w,max_x)
        min_y,max_y = min(y,min_y),max(y+h,max_y)
        roiImg= frame [min_y:max_y,min_x:max_x]
        cv.imshow("roiImg", roiImg)

    return roiImg


def tsr(roiImg,examples_arr):
    try:
        resizedRoi = cv.resize(roiImg, (64, 64))
        sign = cv.inRange(resizedRoi, (89, 91, 149), (255, 255, 255))
        pedistrain= cv.inRange(examples_arr[0], (89, 91, 149), (255, 255, 255))
        no_drive= cv.inRange(examples_arr[1], (89, 91, 149), (255, 255, 255))

        pedistrain_val=0
        no_drive_val=0

        for i in range(64):
            for j in range(64):
                if (sign[i][j]==pedistrain[i][j]):
                    pedistrain_val+=1
                if (sign[i][j]==no_drive[i][j]):
                    no_drive_val+=1

        if pedistrain_val>no_drive_val:
             print("pedistrain")
        else:
            print("no_drive")
    except:
        pass

cap =cv.VideoCapture(0)
pedistrain = cv.resize(cv.imread("pedistrain.png"), (64, 64))
no_drive = cv.resize(cv.imread("noDrive.png"), (64, 64))

while (True):
    ret,frame= cap.read()

    roiImg=tsd(frame)
    examples_arr = [pedistrain, no_drive]
    tsr(roiImg,examples_arr)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
