import cv2
import numpy as np
import math
import heapq

def hough_line_acc(hs, x, y):
    for degrees in range(360):
       theta = math.pi * degrees / 180
       rho = x * math.cos(theta) + y * math.sin(theta)
       if rho >= 0:
           hs[int(rho / 4), degrees] += 1

def hough_lines_acc(edges):
    "Accumulates Hough lines"
    (rows, cols) = edges.shape
    houghSpace= np.zeros((100,360), np.uint8)
    for x in xrange(cols):
        for y in xrange(rows):
            if edges[y, x] != 0:
                hough_line_acc(houghSpace, x, y)
    cv2.normalize(houghSpace)
    return houghSpace

def hough_peaks(hs):
    "Find strongest lines in hough_space"
    heap = []
    (num_rho_buckets, num_degrees) = hs.shape
    for rho in xrange(num_rho_buckets):
        for degree in xrange(num_degrees):
            heapq.heappush(heap, (hs[rho, degree], rho, degree))
    heapq.heapify(heap)
    result = []
    for (value, rho, degree) in heapq.nlargest(10, heap):
        result.append((rho, degree))
    return result

def hough_draw_peaks(hs, peaks):
    for (rho, degree) in peaks:
        cv2.circle(hs, (degree, rho), 4, 255)

def hough_draw_lines(img, lines):
    for (rho_bucket, degree) in peaks:
        # rho = x * cos(theta) + y * sin(theta)
        # ==> x intercept: rho / cos(theta)
        # ==> y intercept: rho / sin(theta)
        # Need to check for 0's
        green = (0, 255, 0)
        theta = math.pi * degree / 180
        rho = (rho_bucket + 0.5) * 4
        s = math.sin(theta)
        c = math.cos(theta)
        print "rho = %f, c = %f, s = %f" % (rho, c, s)
        if (abs(c) < 0.001):
            y = int(rho / s)
            cv2.line(img, (1000, y), (0, y), green)
        elif (abs(s) < 0.001):
            x = int(rho / c)
            cv2.line(img, (x, 0), (x, 1000), green)
        else: 
            x = int(rho / c)
            y = int(rho / s)
            cv2.line(img, (x, 0), (0, y), green)

img = cv2.imread("input/ps1-input0.png")
cv2.startWindowThread()

edges = cv2.Canny(img, 100, 200)
cv2.imshow("ps1-input0", img)
cv2.imshow("Canny Edges", edges)

houghSpace = hough_lines_acc(edges)
cv2.imshow("Hough Space", houghSpace)
cv2.imwrite("output/ps1-2-a-1.png", houghSpace)

peaks = hough_peaks(houghSpace)
hough_draw_peaks(houghSpace, peaks)
cv2.imwrite("output/ps1-2-b-1.png", houghSpace)
cv2.imshow("Hough Space with peaks", houghSpace)

hough_draw_lines(img, peaks)
cv2.imwrite("output/ps1-2-c-1.png", houghSpace)
cv2.imshow("Highlighted Lines", img)

cv2.waitKey()
