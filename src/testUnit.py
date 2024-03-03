import matplotlib.pyplot as plt
import numpy as np
import cv2

win_name = 'scan'
img = cv2.imread('../res/paper02.jpg')
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)

def plotTest():
    x = np.arange(10)
    y = x**2

    plt.subplot(1,2,1) # 1행2열 중 첫번째
    plt.plot(x,y, 'r')

    plt.subplot(1,2,2) # 1행2열 중 두번째
    plt.plot(x, np.sin(x))
    plt.xticks([]) # x좌표 눈금 제거
    
    plt.show()

def key_process():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def affineTest():
    img = cv2.imread('../res/dish01a.jpg')
    rows, cols = img.shape[0:2]
    dx, dy = 100, 50
    mtrx = np.float32([[1,0,dx], [0,1,dy]])
    dst = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy))
    dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,0,0))
    dst3 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
    cv2.imshow('ori', img)
    cv2.imshow('tran',dst)
    cv2.imshow('border_const', dst2)
    cv2.imshow('border_reflect', dst3)

def zoomTest():
    img = cv2.imread('../res/dish01a.jpg')
    height, width = img.shape[0:2]
    smaller = np.float32([[0.5,0,0],[0,0.5,0]])
    bigger = np.float32([[2,0,0],[0,2,0]])
    dst1 = cv2.warpAffine(img, smaller, (int(height*0.5), int(width*0.5))) # by matrix
    dst2 = cv2.warpAffine(img, bigger,(int(height*2), int(width*2)), None, cv2.INTER_AREA)
    dst3 = cv2.resize(img, (int(width*0.7), int(height*0.7)), interpolation=cv2.INTER_AREA)
    dst4 = cv2.resize(img, None, None, 2,2,cv2.INTER_CUBIC) #by rate using resize()
    cv2.imshow('original', img)
    cv2.imshow('smaller', dst1)
    cv2.imshow('bigger', dst2)
    cv2.imshow('size', dst3)
    cv2.imshow('dst4', dst4)

def rotateTest():
    img = cv2.imread('../res/dish01a.jpg')
    rows, cols = img.shape[0:2]
    print(rows, cols)
    center = ((rows/2), (cols/2))
    m30 = cv2.getRotationMatrix2D(center, 90, 0.3)
    img30 = cv2.warpAffine(img, m30, (cols,rows))
    cv2.imshow('rotate', img30)

    d45 = 45 * np.pi / 180
    m45 = np.float32([[np.cos(d45), -1*np.sin(d45), rows//2],
                      [np.sin(d45), np.cos(d45), cols//2]])
    r45 = cv2.warpAffine(img, m45, (cols, rows))
    cv2.imshow('45', r45)

def onMouse(event, x, y, flags, param):
    global pts_cnt
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(draw, (x, y), 10, (0,255,0), -1)
        cv2.imshow(win_name, draw)
        pts[pts_cnt] = [x, y]
        pts_cnt += 1

        if pts_cnt == 4:
            sm = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            TL = pts[np.argmin(sm)]
            BR = pts[np.argmax(sm)]
            TR = pts[np.argmin(diff)]
            BL = pts[np.argmax(diff)]
            pts1 = np.float32([TL, TR, BR, BL])
            w1 = abs(BR[0] - BL[0])
            w2 = abs(TR[0] - TL[0])
            h1 = abs(TR[1] - BR[1])
            h2 = abs(TL[1] - BL[1])
            width = max([w1, w2])
            height = max([h1, h2])

            print(w1, w2, h1, h2, width, height)
            pts2 = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(img, mtrx, [int(width), int(height)])
            cv2.imshow('scanned', result)

def scanTest():
    win_name = 'scan'
    img = cv2.imread('../res/paper02.jpg')
    rows, cols = img.shape[:2]
    draw = img.copy()
    pts_cnt = 0
    pts = np.zeros((4,2), dtype = np.float32)

    cv2.imshow(win_name, img)
    cv2.setMouseCallback(win_name, onMouse)

def findObject():
    img = cv2.imread('../res/check01.png')
    img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntr = contours[0]
    epsilon = 0.05 * cv2.arcLength(cntr, True)
    approx = cv2.approxPolyDP(cntr, epsilon, True)
    cv2.drawContours(img2, [approx], -1, (0,155,110), 2)
    print(cntr)
    cv2.drawContours(img, [cntr], -1, (0,155,110), 2)
    hull = cv2.convexHull(cntr)
    cv2.drawContours(img2, [hull], -1, (0,55,0),1)
    print(cv2.isContourConvex(cntr), cv2.isContourConvex(hull))
    cv2.imshow('gray', gray)
    cv2.imshow('th', th)
    cv2.imshow('dpcontour', img2)

def findSkeleton():
    img = cv2.imread('../res/pose03.jpg', cv2.IMREAD_GRAYSCALE)
    _, biimg = cv2.threshold(img, 157, 255 , cv2.THRESH_BINARY_INV)
    dst = cv2.distanceTransform(biimg, cv2.DIST_L2,5)
    dst = (dst/(dst.max()-dst.min()) * 255).astype(np.uint8)
    skeleton = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -3)
    cv2.imshow('skeleton', skeleton)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3
                                                   ))
    dilated = cv2.dilate(skeleton, k)
    #closing = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, k)
    cv2.imshow('dilated', dilated)

if __name__ == '__main__':
    #plotTest()
    #affineTest()
    #rotateTest()
    #scanTest()
    #findObject()
    findSkeleton()
    key_process()
