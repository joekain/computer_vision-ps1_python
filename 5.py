# VS. 1.py I had to make the Hough space coarser in order to get all the lines to show up.
# This has the effect of collapsing nearby points in hough space into a single bucket which
# removes the effect of some of the noise.

import cv2
import numpy as np
import math
import heapq

num_angle_buckets = 90
max_rho = 0
num_rho_buckets = 0

min_radius = 20
max_radius = 100 
num_radius_buckets = 20

num_dim_buckets = 0
max_dim = 0.0

def init(img):
    global max_rho
    global num_rho_buckets
    (width, height, depth) = img.shape
    max_rho = math.ceil(max(width, height) * math.sqrt(2))
    num_rho_buckets = 2 * int(max_rho / 2)

    global max_dim
    global num_dim_buckets
    max_dim = float(max(width, height))
    num_dim_buckets = max_dim

# Use 0-pi radians.  Using 0-2pi just repeats the same lines in the opposite direction.
def bucket_to_radians(bucket):
    return math.pi * float(bucket) / float(num_angle_buckets)
def radians_to_bucket(angle):
    return int(float(num_angle_buckets) * float(angle) / math.pi)

def bucket_to_rho(bucket):
    return 2 * max_rho * ((bucket) / float(num_rho_buckets) - 0.5)

def rho_to_bucket(rho):
    return int(float(num_rho_buckets) * ((rho / (2 * max_rho)) + 0.5))

def dim_to_bucket(dim):
    return int( num_dim_buckets * float(dim) / max_dim )
def bucket_to_dim(bucket):
    return max_dim * float(bucket) / float(num_dim_buckets)

def radius_to_bucket(radius):
    return int( num_radius_buckets * float(radius - min_radius) / float(max_radius - min_radius))
def bucket_to_radius(bucket):
    return (max_radius - min_radius) * float(bucket) / float(num_radius_buckets) + min_radius

def hough_circle_acc(hs, x, y, r):
    (width, height, depth) = hs.shape
    for bucket in range (num_angle_buckets):
        theta = bucket_to_radians(bucket)
        a = x - r * math.cos(theta)
        b = y + r * math.sin(theta)
        if a > 0 and b > 0:
            hs[dim_to_bucket(a), dim_to_bucket(b), radius_to_bucket(r)] += 1

def hough_circles_acc(edges):
    "Accumulate Hough Circles"
    (rows, cols) = edges.shape
    houghSpace = np.zeros((cols + max_radius, rows + max_radius, num_radius_buckets), np.uint8)
    for radius in xrange(min_radius, max_radius):
        for x in xrange(cols):
            for y in xrange(rows):
                if edges[y, x] != 0:
                    hough_circle_acc(houghSpace, x, y, radius)
    cv2.normalize(houghSpace)
    return houghSpace


def hough_peaks(hs, n):
    "Find strongest lines in hough_space"
    (width, height, depth) = hs.shape
    heap = []
    for radius in xrange(depth):
        for a in xrange(width):
            for b in xrange(height):
                heapq.heappush(heap, (hs[a, b, radius], a, b, radius))
    heapq.heapify(heap)
    result = []
    for (value, a, b, radius) in heapq.nlargest(n, heap):
        result.append((a, b, radius))
    return result

def hough_draw_circles(img, peaks):
    radius = 20
    for (a_bucket, b_bucket, radius_bucket) in peaks:
        a = int(bucket_to_dim(a_bucket))
        b = int(bucket_to_dim(b_bucket))
        radius = int(bucket_to_radius(radius_bucket))
        cv2.circle(img, (a, b), radius, (0, 255, 0))


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

houghSpace = hough_circles_acc(edges)
# cv2.imshow("Hough Space", houghSpace)
# cv2.imwrite("output/ps1-5-c-1.png", houghSpace)

peaks = hough_peaks(houghSpace, 20)
# hough_draw_peaks(houghSpace, peaks)
# cv2.imwrite("output/ps1-5-b-1.png", houghSpace)
# cv2.imshow("Hough Space with peaks", houghSpace)

hough_draw_circles(img, peaks)
cv2.imshow("Highlighted Lines", img)

print "Done"
cv2.waitKey()
