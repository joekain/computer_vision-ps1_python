# VS. 1.py I had to make the Hough space coarser in order to get all the lines to show up.
# This has the effect of collapsing nearby points in hough space into a single bucket which
# removes the effect of some of the noise.

import cv2
import numpy as np
import math
import heapq

num_angle_buckets = 720
max_rho = 0
num_rho_buckets = 0

def init(img):
    global max_rho
    global num_rho_buckets
    (width, height, depth) = img.shape
    max_rho = math.ceil(max(width, height) * math.sqrt(2))
    num_rho_buckets = 2 * int(max_rho / 2)

# Use 0-pi radians.  Using 0-2pi just repeats the same lines in the opposite direction.
def bucket_to_radians(bucket):
    return math.pi * float(bucket) / float(num_angle_buckets)
def radians_to_bucket(angle):
    return int(float(num_angle_buckets) * float(angle) / math.pi)

def bucket_to_rho(bucket):
    return 2 * max_rho * ((bucket) / float(num_rho_buckets) - 0.5)

def rho_to_bucket(rho):
    return int(float(num_rho_buckets) * ((rho / (2 * max_rho)) + 0.5))

def hough_line_acc(hs, x, y):
    for bucket in range(num_angle_buckets):
        theta = bucket_to_radians(bucket)
        rho = x * math.cos(theta) + y * math.sin(theta)
        hs[rho_to_bucket(rho), bucket] += 1

def hough_lines_acc(edges):
    "Accumulates Hough lines"
    (rows, cols) = edges.shape
    houghSpace = np.zeros((num_rho_buckets, num_angle_buckets), np.uint8)
    for x in xrange(cols):
        for y in xrange(rows):
            if edges[y, x] != 0:
                hough_line_acc(houghSpace, x, y)
    cv2.normalize(houghSpace)
    return houghSpace

def hough_peaks(hs, n):
    "Find strongest lines in hough_space"
    heap = []
    for rho in xrange(num_rho_buckets):
        for angle in xrange(num_angle_buckets):
            heapq.heappush(heap, (hs[rho, angle], rho, angle))
    heapq.heapify(heap)
    result = []
    for (value, rho, angle) in heapq.nlargest(n, heap):
        result.append((rho, angle))
    return result

def hough_draw_peaks(hs, peaks):
    for (rho_bucket, angle_bucket) in peaks:
        cv2.circle(hs, (angle_bucket, rho_bucket), 4, 255)

def hough_draw_lines(img, lines):
    for (rho_bucket, angle_bucket) in peaks:
        # rho = x * cos(theta) + y * sin(theta)
        # y = 0 ==> x = rho / cos(theta)
        # x = 0 ==> y = rho / sin(theta)
        # We can draw the line between these points
        # Need to check for 0's
        green = (0, 255, 0)
        theta = bucket_to_radians(angle_bucket)
        rho = bucket_to_rho(rho_bucket)
        s = math.sin(theta)
        c = math.cos(theta)
        if (abs(c) < 0.001):
            y = int(rho / s)
            cv2.line(img, (1000, y), (0, y), green, 1)
        elif (abs(s) < 0.001):
            x = int(rho / c)
            cv2.line(img, (x, 0), (x, 1000), green, 1)
        else:
            x = int(rho / c)
            y = int(rho / s)
            cv2.line(img, (x, 0), (0, y), green, 1)

            # y = 1000 ==> x = (rho - 1000 * sin(theta)) / cos(theta)
            x2 = int((rho - 1000 * s) / c)
            cv2.line(img, (x, 0), (x2, 1000), green, 1)

            # XXX Need to take care of some other cases as well

img = cv2.imread("input/ps1-input1.png")
init(img)
cv2.startWindowThread()
cv2.imshow("ps1-input1", img)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grey Scale ps1-input1", grey)

smooth = grey # I don't think we need smoothing for this image

# smooth = cv2.GaussianBlur(img, (5, 5), 2, 2)
# cv2.imshow("Smoothed", smooth)

edges = cv2.Canny(smooth, 100, 200)
cv2.imshow("Canny Edges", edges)

houghSpace = hough_lines_acc(edges)
cv2.imshow("Hough Space", houghSpace)
cv2.imwrite("output/ps1-4-c-1.png", houghSpace)

peaks = hough_peaks(houghSpace, 10)
hough_draw_peaks(houghSpace, peaks)
cv2.imwrite("output/ps1-4-b-1.png", houghSpace)
cv2.imshow("Hough Space with peaks", houghSpace)

hough_draw_lines(img, peaks)
cv2.imwrite("output/ps1-4-c-1.png", houghSpace)
cv2.imshow("Highlighted Lines", img)

cv2.waitKey()
