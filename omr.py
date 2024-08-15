import cv2
import numpy as np


# --------------------------------------------------- UTILS -------------------------------------------------- #
def Pprocess(img):
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, (lebar, tinggi))
    return img

def RectCont(contours):
    rectCont = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5000:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) == 4:
                rectCont.append(c)
    rectCont = sorted(rectCont, key=cv2.contourArea, reverse=True)
    return rectCont

def CornerPoint(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    return approx

def Reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def Boxes(img):
    rows = np.vsplit(img, 15)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 4)
        for box in cols:
            boxes.append(box)
    return boxes

# ------------------------------------------------ RESIZE ---------------------------------------------------- #
lebar, tinggi = 600, 600

# --------------------------------------- MAIN FUNCTION ------------------------------------------------------ #
def OMR(img, ans1_15, ans16_30):
    img = Pprocess(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    edge = cv2.Canny(blur, 338, 0)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    rectCont = RectCont(contours)
    cont1_15 = CornerPoint(rectCont[0])
    cont16_30 = CornerPoint(rectCont[1])

    if cont1_15.size != 0 and cont16_30.size != 0:
        cv2.drawContours(img, cont1_15, -1, (0, 255, 0), 10)
        cv2.drawContours(img, cont16_30, -1, (0, 255, 0), 10)

        cont1 = Reorder(cont1_15)
        cont2 = Reorder(cont16_30)

        cont1_pt1 = np.float32(cont1)
        cont1_pt2 = np.float32([[0, 0], [200, 0], [0, tinggi], [200, tinggi]])
        matrix1 = cv2.getPerspectiveTransform(cont1_pt1, cont1_pt2)
        imgwrap1 = cv2.warpPerspective(img, matrix1, (200, tinggi))

        cont2_pt1 = np.float32(cont2)
        cont2_pt2 = np.float32([[0, 0], [200, 0], [0, tinggi], [200, tinggi]])
        matrix2 = cv2.getPerspectiveTransform(cont2_pt1, cont2_pt2)
        imgwrap2 = cv2.warpPerspective(img, matrix2, (200, tinggi))

        imgwrapgray1 = cv2.cvtColor(imgwrap1, cv2.COLOR_BGR2GRAY)
        imgThresh1 = cv2.threshold(imgwrapgray1, 140, 255, cv2.THRESH_BINARY_INV)[1]

        imgWarpGray2 = cv2.cvtColor(imgwrap2, cv2.COLOR_BGR2GRAY)
        imgThresh2 = cv2.threshold(imgWarpGray2, 140, 255, cv2.THRESH_BINARY_INV)[1]

        boxes1 = Boxes(imgThresh1)
        boxes2 = Boxes(imgThresh2)

        valpx1 = np.zeros((15, 4))
        countC1 = 0
        countR1 = 0
        for image in boxes1:
            totalpx = cv2.countNonZero(image)
            valpx1[countR1][countC1] = totalpx
            countC1 += 1
            if countC1 == 4:
                countR1 += 1
                countC1 = 0

        valpx2 = np.zeros((15, 4))
        countC2 = 0
        countR2 = 0
        for image in boxes2:
            totalpx = cv2.countNonZero(image)
            valpx2[countR2][countC2] = totalpx
            countC2 += 1
            if countC2 == 4:
                countR2 += 1
                countC2 = 0

        myindex1 = []
        for x in range(0, 15):
            arr = valpx1[x]
            myindexval = np.where(arr == np.amax(arr))
            myindex1.append(myindexval[0][0])

        myindex2 = []
        for x in range(0, 15):
            arr = valpx2[x]
            myindexval = np.where(arr == np.amax(arr))
            myindex2.append(myindexval[0][0])

        n1_15 = [(1 if ans1_15[x] == myindex1[x] else 0) for x in range(15)]
        n16_30 = [(1 if ans16_30[x] == myindex2[x] else 0) for x in range(15)]

        total_nilai = n1_15 + n16_30
        total_score = (sum(total_nilai) / 30) * 100

        return img, total_score, n1_15, n16_30

    return None, None, None, None
