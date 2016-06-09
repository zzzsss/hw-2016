# provide the algorithm for weak one-barrier
from helper import *
import copy
import numpy as np
from math import sqrt
from subprocess import Popen, PIPE
import os

# !! always return new ones and always assuming sorted by x

# def weak_one(plist, L, r):
#     # just testing
#     l = len(plist)
#     d = L / (l+1)
#     return [[d*(i+1), z[1]] for i, z in enumerate(plist)]
#     # return copy.deepcopy(plist)

def weak_one(plist, L, r):
    # print(plist)
    # write file
    with open("in", 'w') as f:
        f.write("%d %d\n" % (L, r))
        for p in plist:
            f.write("%f %f\n" % (p[0], p[1]))
    # execute
    if not os.path.exists("a.exe"):
        os.system("g++ -std=c++11 code.cpp")
    p = Popen("a.exe", shell=True, stdout=PIPE, stderr=PIPE)
    # p.wait()
    out = p.stdout.readlines()
    # print(out)
    # return
    ret = copy.deepcopy(plist)
    xlist = [float(z) for z in out[0].split()]
    for i, p in enumerate(ret):
        p[0] = xlist[i]
    return ret

def weak_k(plist, L, r, k, policy):
    p = [valuer_random, value_x(r)][policy]
    indexes = split_k(plist, k, p)
    retlist = copy.deepcopy(plist)
    for one in indexes:
        # print(one)
        oneafter = weak_one([plist[z] for z in one], L, r)
        for z, v in zip(one, oneafter):
            retlist[z] = v
    return retlist, indexes

def strong_one(plist, L, r, post):
    r += 0.0001  # for float tolerance
    retlist = weak_one(plist, L, r)
    # print(retlist)
    yvalue = np.average([z[1] for z in plist])
    for one in retlist:
        one[1] = yvalue
    if post == 0:
        return retlist
    # simple post processing
    for i in range(1, len(plist)-1):
        # skip the first and last one for simplicity
        left_y = retlist[i-1][1]
        right_y = retlist[i+1][1]   # currently should only be yvalue
        # calculate the span
        left_dx = retlist[i][0] - retlist[i-1][0]
        right_dx = retlist[i+1][0] - retlist[i][0]
        # print(r, left_dx, right_dx)
        # print(2*r, right_dx)
        left_dy = sqrt((2*r)**2 - left_dx**2)
        right_dy = sqrt((2*r)**2 - right_dx**2)
        left_up, left_down = left_y+left_dy, left_y-left_dy
        right_up, right_down = right_y+right_dy, right_y-right_dy
        mup = min(left_up, right_up)
        mdown = max(left_down, right_down)
        # move the y
        new_y = plist[i][1]
        # print(i, new_y, mup, mdown, right_up, right_down)
        if mup < mdown:
            # impossible, but really nothing is impossible
            new_y = retlist[i][1]
        else:
            new_y = min(new_y, mup)
            new_y = max(new_y, mdown)
        retlist[i][1] = new_y
    return retlist

def strong_k(plist, L, r, k, policy, weight, post):
    p = [valuer_random, valuer_y, valuer_xy(r, weight)][policy]
    indexes = split_k(plist, k, p)
    retlist = copy.deepcopy(plist)
    for one in indexes:
        oneafter = strong_one([plist[z] for z in one], L, r, post)
        for z, v in zip(one, oneafter):
            retlist[z] = v
    return retlist, indexes
