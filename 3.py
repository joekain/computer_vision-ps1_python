# VS. 1.py I had to make the Hough space coarser in order to get all the lines to show up.
# This has the effect of collapsing nearby points in hough space into a single bucket which
# removes the effect of some of the noise.

import cv2
import numpy as np
import math
import heapq

num_degree_buckets = 180
max_rho = math.ceil(math.sqrt(2) * 256) 
num_rho_buckets = int(max_rho / 2)

def bucket_to_degree(bucket):
    return int(360.0 * (bucket) / float(num_degree_buckets))

def bucket_to_rho(bucket):
    return int(max_rho * (bucket) / float(num_rho_buckets))

def rho_to_bucket(rho):
    return int(float(num_rho_buckets) * rho / max_rho); 

def hough_line_acc(hs, x, y):
    for bucket in range(num_degree_buckets):
        degrees = bucket_to_degree(bucket)
        theta = math.pi * degrees / 180.0;
        rho = x * math.cos(theta) + y * math.sin(theta)
        if rho >= 0:
            hs[rho_to_bucket(rho), bucket] += 1

def hough_lines_acc(edges):
    "Accumulates Hough lines"
    (rows, cols) = edges.shape
    houghSpace= np.zeros((num_rho_buckets, num_degree_buckets), np.uint8)
    for x in xrange(cols):
        for y in xrange(rows):
            if edges[y, x] != 0:
                hough_line_acc(houghSpace, x, y)
    cv2.normalize(houghSpace)
    return houghSpace

def hough_peaks(hs):
    "Find strongest lines in hough_space"
    heap = []
    for rho in xrange(num_rho_buckets):
        for degree in xrange(num_degree_buckets):
            heapq.heappush(heap, (hs[rho, degree], rho, degree))
    heapq.heapify(heap)
    result = []
    for (value, rho, degree) in heapq.nlargest(10, heap):
        print "value = %f at (%d, %d)" % (value, rho, degree)
        result.append((rho, degree))
    return result

def hough_draw_peaks(hs, peaks):
    for (rho_bucket, degree_bucket) in peaks:
        cv2.circle(hs, (degree_bucket, rho_bucket), 4, 255)

def hough_draw_lines(img, lines):
    for (rho_bucket, degree_bucket) in peaks:
        # rho = x * cos(theta) + y * sin(theta)
        # ==> x intercept: rho / cos(theta)
        # ==> y intercept: rho / sin(theta)
        # Need to check for 0's
        green = (0, 255, 0)
        degree = bucket_to_degree(degree_bucket)
        theta = math.pi * degree / 180
        rho = bucket_to_rho(rho_bucket)
        s = math.sin(theta)
        c = math.cos(theta)
        # print "rho = %f, theta = %f, c = %f, s = %f" % (rho, theta, c, s)
        print "rho_bucket = %d degree = %d" % (rho_bucket, degree)
        if (abs(c) < 0.001):
            y = int(rho / s)
            cv2.line(img, (1000, y), (0, y), green), 3
        elif (abs(s) < 0.001):
            x = int(rho / c)
            cv2.line(img, (x, 0), (x, 1000), green, 3)
        else: 
            x = int(rho / c)
            y = int(rho / s)
            cv2.line(img, (x, 0), (0, y), green, 3)

img = cv2.imread("input/ps1-input0-noise.png")
cv2.startWindowThread()
cv2.imshow("ps1-input0-noise", img)

smooth = cv2.GaussianBlur(img, (5, 5), 2, 2)
cv2.imshow("Smoothed", smooth)

edges = cv2.Canny(smooth, 100, 200)
cv2.imshow("Canny Edges", edges)

houghSpace = hough_lines_acc(edges)
cv2.imshow("Hough Space", houghSpace)
cv2.imwrite("output/ps1-3-a-1.png", houghSpace)

peaks = hough_peaks(houghSpace)
hough_draw_peaks(houghSpace, peaks)
cv2.imwrite("output/ps1-3-b-1.png", houghSpace)
cv2.imshow("Hough Space with peaks", houghSpace)

hough_draw_lines(img, peaks)
cv2.imwrite("output/ps1-3-c-1.png", houghSpace)
cv2.imshow("Highlighted Lines", img)

cv2.waitKey()
