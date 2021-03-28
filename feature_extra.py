import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import collections
import csv
from PIL import Image
from skimage.draw import ellipse
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
from skimage.feature import greycomatrix, greycoprops

class cal_feature():
    def __init__(self,img,ribbon_length):
        self.img = img
        self.length_ribbon_of_pixels = ribbon_length

    # 计算方位角函数
    def calc_angle(self, x1, y1, x2, y2):
        angle = 0
        dy = y2 - y1
        dx = x2 - x1
        if dx == 0 and dy > 0:
            angle = 0
        if dx == 0 and dy < 0:
            angle = 180
        if dy == 0 and dx > 0:
            angle = 90
        if dy == 0 and dx < 0:
            angle = 270
        if dx > 0 and dy > 0:
            angle = math.atan(dx / dy) * 180 / math.pi
        elif dx < 0 and dy > 0:
            angle = 360 + math.atan(dx / dy) * 180 / math.pi
        elif dx < 0 and dy < 0:
            angle = 180 + math.atan(dx / dy) * 180 / math.pi
        elif dx > 0 and dy < 0:
            angle = 180 + math.atan(dx / dy) * 180 / math.pi
        return angle

    def intermediates(self, p1, p2, nb_points=8):
        """"Return a list of nb_points equally spaced points
        between p1 and p2"""
        # If we have 8 intermediate points, we have 8+1=9 spaces
        # between p1 and p2
        x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
        y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

        return [[p1[0] + i * x_spacing, p1[1] + i * y_spacing]
                for i in range(1, nb_points + 1)]

    def perpen45(self, mid_point, length):  # \ lines
        L1 = round(length / 2)
        L2 = round(length / 2)
        # left > mid
        while (L1 > 0):
            self.B[mid_point[0] - (1 * L1), mid_point[1] - (1 * L1)] = 255
            L1 = L1 - 1
        # mid > right
        while (L2 > 0):
            self.B[mid_point[0] + (1 * L2), mid_point[1] + (1 * L2)] = 255
            L2 = L2 - 1

    def perpen135(self, mid_point, length):  # \ lines
        L1 = round(length / 2)
        L2 = round(length / 2)
        # left > mid
        while (L1 > 0):
            self.B[mid_point[0] - (1 * L1), mid_point[1] + (1 * L1)] = 255
            L1 = L1 - 1
        # mid > right
        while (L2 > 0):
            self.B[mid_point[0] + (1 * L2), mid_point[1] - (1 * L2)] = 255
            L2 = L2 - 1

    def ribbon_pixels(self, mid_point, ang, length):
        if (ang == 0):
            plt.plot([mid_point[1], mid_point[1]], [mid_point[0] + 20, mid_point[0] - 20], color='#000000',
                     label='line 1', linewidth=1)
            self.B[mid_point[0] - 20:mid_point[0] + 20, mid_point[1]] = 255
            # B[ mid_point[0]-20:mid_point[0]+20,mid_point[1]]= B[ mid_point[0]-20:mid_point[0]+20,mid_point[1]] +60
        elif (ang == 90):
            plt.plot([mid_point[1] - 30, mid_point[1] + 30], [mid_point[0], mid_point[0]], color='#000000',
                     label='line 1', linewidth=1)
            self.B[mid_point[0], mid_point[1] - 20:mid_point[1] + 20] = 255
            # B[mid_point[0],mid_point[1]-30:mid_point[1]+30] = B[mid_point[0],mid_point[1]-30:mid_point[1]+30] +60
        elif (ang == 135):
            plt.plot([mid_point[1] + 20, mid_point[1] - 20], [mid_point[0] - 20, mid_point[0] + 20], color='#000000',
                     label='line 1', linewidth=1)
            self.perpen135(mid_point, 20)
        elif (ang == 45):
            plt.plot([mid_point[1] - 20, mid_point[1] + 20], [mid_point[0] - 20, mid_point[0] + 20], color='#000000',
                     label='line 1', linewidth=1)
            self.perpen45(mid_point, 20)

    def Enp(self, coMat):
        sum = 0
        for _ in coMat:
            for i in _:
                if (i != 0):
                    sum = sum + (i * math.log2(i))
        return -sum
    def main(self):
        #image = cv2.imread('./bio/test5.jpg')
        # tt = image[500:650, 580:780,0]  #test3
        # tt = image[390:750,400:720,0]  #test4
        #tt = image[630:770, 270:400, 0]  # test5
        # tt = image[220:880,120:720,0]

        # image = cv2.imread('./bio/2.jpg')
        # tt = image[:,:,1]
        tt = self.img.copy()
        t2 = tt.copy()
        edged = cv2.Canny(t2, 60, 200)
        contours, hierarchy = cv2.findContours(edged,
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        """
        cv2.imshow('Canny Edges After Contouring', edged)
        cv2.waitKey(0)
        """

        # -1 signifies drawing all contours
        cv2.drawContours(t2, contours, -1, (0, 255, 0), 2)
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        """
        cv2.imshow('Contours', t2)
        #cv2.imshow('Contours', image)
        cv2.waitKey(0)
        """

        fig, (ax2) = plt.subplots(ncols=1, figsize=(9, 4))

        # find contour in ROIs                                          -step 1
        plt.gray()
        # ax2.imshow(tt)
        list_len = []
        for contour in find_contours(t2, 5):
            list_len.append(len(contour))
        list_len.sort()

        # polygonal approximation  methods                              -step 2
        for contour in find_contours(t2, 5):
            # print(contour)
            coords = approximate_polygon(contour, tolerance=5)
            if (len(contour) == list_len[-2]):
                ax2.plot(coords[:, 1], coords[:, 0], '-r', linewidth=0.5)
                polygon_app = coords

            # print("Number of coordinates:", len(contour), len(coords))
        plt.imshow(tt)
        ax2.axis((0, 700, 0, 700))
        plt.show()
        # just show the polygon model
        #plt.plot(polygon_app[:,1],polygon_app[:,0])
        #plt.show()

        # method 1 , use polygonal boundary and calculate the boundary's point then to find norm
        """
        print(intermediates(boundary[1],boundary[2]))
        x = []
        x.append(boundary[0].tolist())
        for p in intermediates(boundary[0],boundary[1]):
            x.append(p)
        x.append(boundary[1].tolist())
        """
        # calculate the angle in 2 inflection
        # method 2 , use polygonal point to find inflection in original image
        h, w = tt.shape
        self.B = np.zeros((h, w))  # mass boundary image
        count = 0
        bnd = []  # boundary points vector
        for i in range(h):
            for j in range(w):
                if (tt[i, j] > 240):
                    self.B[i, j] = 255
                    bnd.append([i, j])  # boundary list
                    count = count + 1  # boundary count
        # plt.imshow(B)
        # for gg in bnd:      # show inflction on boundary
        #    plt.scatter(gg[1], gg[0], s=75, alpha=.5)
        # plt.show()

        inf = []  # record inflection on nearest boundary points          -step 3
        for p in polygon_app:
            min_d = max(h, w)
            for b in bnd:
                D = math.sqrt(math.pow(p[0] - b[0], 2) + math.pow(p[1] - b[1], 2))
                if (D < min_d):
                    min_d = D
                    tmp = b
            inf.append(tmp)
            # print(P)
            # print(min_d)
        # plt.imshow(B)
        c = 0

        plt.imshow(tt)
        for gg in inf:  # show inflction on boundary
            c = c + 1
            plt.scatter(gg[1], gg[0], s=75, alpha=.5)
            # if(c==4):
            #    break
        plt.show()

        # print("inflection point", inf)

        Angle = []
        # find all points between 2 inflection point (P)                -step 4
        for i in range(len(inf) - 1):
            a = self.calc_angle(inf[i][0], inf[i][1], inf[i + 1][0], inf[i + 1][1])
            if ((a > 22.5 and a < 67.5) or (a > 202.5 and a < 247.5)):
                # Angle.append(45)
                Angle.append(135)
            elif ((a > 67.5 and a < 112.5) or (a > 247.5 and a < 292.5)):
                Angle.append(90)
            elif ((a > 112.5 and a < 157.5) or (a > 292.5 and a < 337.5)):
                # Angle.append(135)
                Angle.append(45)
            elif ((a > 157.5 and a < 202.5) or (a > 337.5 or a < 22.5)):
                Angle.append(0)
        # print(Angle)

        # find all points between 2 inflection point (P)                -step 5
        x_count = 0;
        y_count = 0;
        G = []
        a1 = 8
        a2 = 9
        i = 0
        N = 0  # number of all boundary points
        #length_ribbon_of_pixels = 4
        aa = []  # ribbon points of one boundary point
        bb = []

## ===============================
        for a1 in range(len(inf) - 1):
            a2 = a1 + 1
            tmp1 = []
            tmp2 = []
            tmp1.append(inf[a1])
            tmp2.append(inf[a1])
            for b in bnd:
                if (a2 < len(inf) / 2):
                    # y (up to down) --x1--
                    #                --x2--
                    if (b[0] <= inf[a1][0] and b[0] >= inf[a2][0] and b[1] <= max(inf[a1][1], inf[a2][1])):
                        y_count = y_count + 1  # get all the point of x where in range(P1,P2)
                        tmp1.append([b[0], b[1]])
                        # x (left to right) |x1    x2|
                        if (b[1] <= inf[a1][1] and b[1] >= inf[a2][1] and b[0] >= min(inf[a1][0], inf[a2][0])):
                            x_count = x_count + 1  # get all the point of y where in range(P1,P2)
                            tmp2.append([b[0], b[1]])
                else:
                    # y (up to down) --x1--
                    #                --x2--
                    if (b[0] >= inf[a1][0] and b[0] <= inf[a2][0] and b[1] >= min(inf[a1][1], inf[a2][1])):
                        y_count = y_count + 1  # get all the point of x where in range(P1,P2)
                        tmp1.append([b[0], b[1]])
                        # x (left to right) |x1    x2|
                        if (b[1] >= inf[a1][1] and b[1] <= inf[a2][1] and b[0] <= max(inf[a1][0], inf[a2][0])):
                            x_count = x_count + 1  # get all the point of y where in range(P1,P2)
                            tmp2.append([b[0], b[1]])

            if (x_count < y_count):
                G = tmp1
            else:
                G = tmp2

            # show the get point whether on bounary in range(P1,P2)
            plt.imshow(tt)

            k = 0

            for gg in G:  # show inflction on boundary
                self.ribbon_pixels(gg, Angle[i], 0)
                aa = []
                if (Angle[i] == 0):
                    # plt.plot([gg[1], gg[1]], [gg[0] + self.length_ribbon_of_pixels, gg[0] - self.length_ribbon_of_pixels], 'g-', label='line 1', linewidth=1)
                    for k in range(self.length_ribbon_of_pixels * 2):
                        aa.append([gg[1], (gg[0] - self.length_ribbon_of_pixels) + k])
                    bb.append(aa)
                    N = N + 1

                elif (Angle[i] == 90):
                    # plt.plot([gg[1] - self.length_ribbon_of_pixels, gg[1] + self.length_ribbon_of_pixels], [gg[0], gg[0]], 'g-', label='line 1', linewidth=1)
                    for k in range(self.length_ribbon_of_pixels * 2):
                        aa.append([(gg[1] - self.length_ribbon_of_pixels) + k, gg[0]])
                    bb.append(aa)
                    N = N + 1

                elif (Angle[i] == 135):
                    # plt.plot([gg[1] + self.length_ribbon_of_pixels, gg[1] - self.length_ribbon_of_pixels], [gg[0] - self.length_ribbon_of_pixels, gg[0] + self.length_ribbon_of_pixels], 'g-', label='line 1', linewidth=1)
                    for k in range(self.length_ribbon_of_pixels * 2):
                        aa.append([(gg[1] + self.length_ribbon_of_pixels) - k, (gg[0] - self.length_ribbon_of_pixels) + k])
                    bb.append(aa)
                    N = N + 1

                elif (Angle[i] == 45):
                    # plt.plot([gg[1] - self.length_ribbon_of_pixels, gg[1] + self.length_ribbon_of_pixels], [gg[0] - self.length_ribbon_of_pixels, gg[0] + self.length_ribbon_of_pixels], 'g-', label='line 1', linewidth=1)
                    for k in range(self.length_ribbon_of_pixels * 2):
                        aa.append([(gg[1] - self.length_ribbon_of_pixels) + k, (gg[0] - self.length_ribbon_of_pixels) + k])
                    bb.append(aa)
                    N = N + 1

            i = i + 1

        plt.show()
        ## ===============================
        # calculate gradient features                                    -step 6
        gradient_d = []
        i = 0
        j = 0

        # rms gradient feature
        for i in range(len(bb)):
            temp = 0
            for j in range(self.length_ribbon_of_pixels - 1):
                temp = temp + np.square(abs(tt[bb[i][j][1], bb[i][j][0]] - tt[bb[i][j + 1][1], bb[i][j + 1][0]]))
            gradient_d.append(np.sqrt(temp / self.length_ribbon_of_pixels))

        maxVal = 0
        minVal = 256
        # max pixel of ribbon of pixels
        for i in range(len(bb)):
            for j in range(self.length_ribbon_of_pixels):
                if (tt[bb[i][j][1], bb[i][j][0]] > maxVal):
                    maxVal = tt[bb[i][j][1], bb[i][j][0]]

        # min pixel of ribbon of pixels
        for i in range(len(bb)):
            for j in range(self.length_ribbon_of_pixels):
                if (tt[bb[i][j][1], bb[i][j][0]] < minVal):
                    minVal = tt[bb[i][j][1], bb[i][j][0]]

        # modified gradient feature
        Adg = sum(gradient_d) / (N * (maxVal - minVal))

        ################################################

        # Coefficient of Variation of Gradient Strength
        M = 5
        Variance = []
        Gcv = []

        for i in range(len(bb)):
            temp = 0
            window_mean = 0
            up2 = 0
            up1 = 0
            down2 = 0
            down1 = 0
            k = 0
            l = 0
            maxVal = 0
            window = []
            tmp = 0
            for j in range(self.length_ribbon_of_pixels):

                if (tt[bb[i][j + 2][1], bb[i][j + 2][0]] == []):
                    up2 = tt[bb[i][j][1], bb[i][j][0]]

                elif (tt[bb[i][j + 1][1], bb[i][j + 1][0]] == []):
                    up1 = tt[bb[i][j][1], bb[i][j][0]]

                elif (tt[bb[i][j - 2][1], bb[i][j - 2][0]] == []):
                    down2 = tt[bb[i][j][1], bb[i][j][0]]

                elif (tt[bb[i][j - 1][1], bb[i][j - 1][0]] == []):
                    down1 = tt[bb[i][j][1], bb[i][j][0]]
                else:
                    up2 = tt[bb[i][j + 2][1], bb[i][j + 2][0]]
                    up1 = tt[bb[i][j + 1][1], bb[i][j + 1][0]]
                    down2 = tt[bb[i][j - 2][1], bb[i][j - 2][0]]
                    down1 = tt[bb[i][j - 1][1], bb[i][j - 1][0]]

                window_mean = (tt[bb[i][j][1], bb[i][j][0]] + up2 + up1 + down2 + down1) / M
                window.append(np.square(tt[bb[i][j][1], bb[i][j][0]] - window_mean))

            for l in range(self.length_ribbon_of_pixels):

                if (l == (self.length_ribbon_of_pixels - 1)):
                    up1 = window[l]
                    up2 = window[l]
                    down2 = window[l - 2]
                    down1 = window[l - 1]

                elif (l == (self.length_ribbon_of_pixels - 2)):
                    up2 = window[l]
                    up1 = window[l + 1]
                    down2 = window[l - 2]
                    down1 = window[l - 1]

                elif (l == 0):
                    down1 = window[l]
                    down2 = window[l]
                    up2 = window[l + 2]
                    up1 = window[l + 1]

                elif (l == 1):
                    down2 = window[l]
                    up2 = window[l + 2]
                    up1 = window[l + 1]
                    down1 = window[l - 1]

                else:
                    up2 = window[l + 2]
                    up1 = window[l + 1]
                    down2 = window[l - 2]
                    down1 = window[l - 1]

                tmp = (window[l] + up2 + up1 + down2 + down1) / M
                if (tmp > maxVal):
                    maxVal = tmp

                Variance.append(tmp)
            Gcv.append(maxVal)
        Gcv_total = sum(Gcv)
        Gcv_total_mean = Gcv_total / N


        #plt.imshow(self.B)
        #plt.show()


        # calculate texture features                                    -step 7
        R = np.count_nonzero(self.B == 255)  # total numbers of pixel pairs in ROI
        self.B = self.B.astype(int)
        # co-orrurrence matrices    P(i,j)
        result = greycomatrix(self.B, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
        CoMat = np.zeros((256, 256, 1, 4), dtype=float)
        CoMat[:, :, 0, 0] = result[:, :, 0, 0] / (np.sum(result[:, :, 0, 0]))  # normalize matrix
        CoMat[:, :, 0, 1] = result[:, :, 0, 1] / (np.sum(result[:, :, 0, 1]))  # normalize matrix
        CoMat[:, :, 0, 2] = result[:, :, 0, 2] / (np.sum(result[:, :, 0, 2]))  # normalize matrix
        CoMat[:, :, 0, 3] = result[:, :, 0, 3] / (np.sum(result[:, :, 0, 3]))  # normalize matrix

        # Entropy

        E_0 = self.Enp(CoMat[:, :, 0, 0])
        E_45 = self.Enp(CoMat[:, :, 0, 1])
        E_90 = self.Enp(CoMat[:, :, 0, 2])
        E_135 = self.Enp(CoMat[:, :, 0, 3])
        # second Momenet //ASM

        ASM_0 = greycoprops(CoMat, 'ASM')[0, 0]
        ASM_45 = greycoprops(CoMat, 'ASM')[0, 1]
        ASM_90 = greycoprops(CoMat, 'ASM')[0, 2]
        ASM_135 = greycoprops(CoMat, 'ASM')[0, 3]
        # different moment // contrast

        DM_0 = greycoprops(CoMat, 'contrast')[0, 0]
        DM_45 = greycoprops(CoMat, 'contrast')[0, 1]
        DM_90 = greycoprops(CoMat, 'contrast')[0, 2]
        DM_135 = greycoprops(CoMat, 'contrast')[0, 3]
        # inverse different

        INV_0 = greycoprops(CoMat, 'homogeneity')[0, 0]
        INV_45 = greycoprops(CoMat, 'homogeneity')[0, 1]
        INV_90 = greycoprops(CoMat, 'homogeneity')[0, 2]
        INV_135 = greycoprops(CoMat, 'homogeneity')[0, 3]
        # correlation

        Corr_0 = greycoprops(CoMat, 'correlation')[0, 0]
        Corr_45 = greycoprops(CoMat, 'correlation')[0, 1]
        Corr_90 = greycoprops(CoMat, 'correlation')[0, 2]
        Corr_135 = greycoprops(CoMat, 'correlation')[0, 3]

        #print("Entropy = ", E_0, E_45, E_90, E_135)
        #print("second Moment = ", ASM_0, ASM_45, ASM_90, ASM_135)
        #print("different moment = ", DM_0, DM_45, DM_90, DM_135)
        #print("inverse different = ", INV_0, INV_45, INV_90, INV_135)
        #print("correlation = ", Corr_0, Corr_45, Corr_90, Corr_135)

        return [E_0, E_45, E_90, E_135, ASM_0, ASM_45, ASM_90, ASM_135, DM_0, DM_45, DM_90, DM_135, INV_0, INV_45, INV_90, INV_135, Corr_0, Corr_45, Corr_90, Corr_135,Adg,Gcv_total_mean]
