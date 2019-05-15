# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 当鼠标按下时变为True
drawing = False
# 如果mode为true绘制矩形。按下'm' 变成绘制曲线。
mode = True
ix, iy = -1, -1
contour = []
# lsPointsChoose = []  # 用于转化为darry 提取多边形ROI
# tpPointsChoose = []  # 用于画点

# 写好回调函数
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        contour.append((x, y))
        cv2.circle(img, (x, y), 1, (0, 255, 0))
        # lsPointsChoose.append([x, y])  # 用于转化为darry 提取多边形ROI
        # tpPointsChoose.append((x, y))  # 用于画点
        # print(len(tpPointsChoose))
        # for i in range(len(tpPointsChoose) - 1):
        #     print('i', i)
        #     cv2.line(img, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)


def nothing(event, x, y, flags, param):
    pass
    # print "[",x,",",y,"]"


# 创建图像与窗口ii
# 将窗口和回调函数邦定W
img_num = 18
img_path = "/Users/wangxiaoman/Desktop/research/imgcrop-master/example/train/"
thre = 20

img = np.zeros((512, 512, 3), np.uint8)
img = cv2.imread(img_path + "{}.jpg".format(img_num))
cv2.namedWindow("Mouse_draw")
cv2.setMouseCallback("Mouse_draw", draw_circle)

while (1):

    cv2.imshow("Mouse_draw", img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord("n"):
        img_num += 1
        img = cv2.imread(img_path + "{}.jpg".format(img_num))
        contour = []
    if k == ord("p"):
        img_num -= 1
        img = cv2.imread(img_path + "{}.jpg".format(img_num))
        contour = []
    if k == ord("a"):
        cv2.setMouseCallback("Mouse_draw", nothing)
        oriImg = cv2.imread(img_path + "{}.jpg".format(img_num))
        contour = np.array(contour)
        # print(contour)

        color1 = oriImg[contour[0][1]][contour[0][0]][0]
        color2 = oriImg[contour[1][1]][contour[1][0]][0]
        diff = abs(int(color1) - int(color2)) - thre
        # diff = int(abs(int(color1) - int(color2)) * thre)
        contour = np.reshape(contour[2:, :], (-1, 1, 2))

        result = np.zeros_like(img)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if abs(int(oriImg[j][i][0]) - int(color1)) > diff and cv2.pointPolygonTest(contour, (i, j), False) > 0.:
                    result[j][i][0] = oriImg[j][i][0]
                    result[j][i][1] = oriImg[j][i][1]
                    result[j][i][2] = oriImg[j][i][2]

        while (1):
            img_appended = np.append(oriImg, result, axis=1)
            cv2.imshow("Mouse_draw", img_appended)
            k = cv2.waitKey(20)
            if k == ord("c"):
                cv2.imwrite(img_path + "{}_mask.jpg".format(img_num), result)
                img_num += 1
                img = cv2.imread(img_path + "{}.jpg".format(img_num))
                contour = []
                cv2.setMouseCallback("Mouse_draw", draw_circle)
                break
            # if k == ord("n"):
            # img_num+=1
            # img = cv2.imread("/Users/liujiajun/Downloads/breast_cancer/image{}.jpg".format(img_num))
            # break
            if k == ord("s"):
                img = cv2.imread(img_path + "{}.jpg".format(img_num))
                contour = []
                cv2.setMouseCallback("Mouse_draw", draw_circle)
                break
        # cv2.imwrite("/Users/liujiajun/Downloads/breast_cancer/result.jpg",result)

    if k == ord("b"):
        break;

cv2.destroyAllWindows()
