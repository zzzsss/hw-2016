#!/bin/python3

import sys, random
from basic import *
from math import sqrt

# main procedure
def main():
    # set the configs
    configs = {"L":100, "rr":1.5, "L-W":2, "L-r":10, "k":1, "weak":1, "times":1, "policy":0, "weight":0.1, "simplepost": 0, "random":12345}
    for s in sys.argv[1:]:
        tmp_l = s.split(":")
        assert len(tmp_l)==2
        configs[tmp_l[0]] = tmp_l[1]
    # some reports
    random.seed(int(configs['random']))     # ensure same
    print(configs)
    real_configs = dict()
    real_configs["p_L"] = float(configs["L"])
    real_configs["p_k"] = int(configs["k"])
    real_configs["p_W"] = real_configs["p_L"] / float(configs["L-W"])
    real_configs["p_r"] = real_configs["p_L"] / float(configs["L-r"])
    real_configs["p_n"] = int(float(configs["rr"]) * real_configs["p_L"] * real_configs["p_k"] / (2 * real_configs["p_r"]) + 0.999)
    real_configs["p_weak"] = bool(int(configs["weak"]))
    real_configs["p_times"] = int(configs["times"])
    real_configs["policy"] = int(configs["policy"])
    real_configs["weight"] = float(configs["weight"])
    real_configs["simplepost"] = int(configs["simplepost"])
    print(real_configs)
    # execute for "times" time
    ret = []
    pindexes = [[i for i in range(real_configs["p_n"])]]
    for i in range(real_configs["p_times"]):
        plist = random_init(real_configs["p_n"], real_configs["p_L"], real_configs["p_W"])
        if real_configs["p_weak"]:
            if real_configs["p_k"] > 1:
                pfinal, pindexes = weak_k(plist, real_configs["p_L"], real_configs["p_r"], real_configs["p_k"], real_configs["policy"])
            else:
                pfinal = weak_one(plist, real_configs["p_L"], real_configs["p_r"])
        else:
            if real_configs["p_k"] > 1:
                pfinal, pindexes = strong_k(plist, real_configs["p_L"], real_configs["p_r"], real_configs["p_k"], real_configs["policy"], real_configs["weight"], real_configs["simplepost"])
            else:
                pfinal = strong_one(plist, real_configs["p_L"], real_configs["p_r"], real_configs["simplepost"])
        ret.append([plist, pfinal])
    # analysis and possibly draw
    dis = 0.0
    for pair in ret:
        for p1, p2 in zip(pair[0], pair[1]):
            dis += sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    dis /= len(ret)
    dis /= real_configs["p_n"]
    # print(ret)
    print("Average distance is: %s and avg/r is: %s" % (dis, dis/real_configs["p_r"]))
    if real_configs["p_times"] == 1:
        draw_one(ret[0][0], ret[0][1], real_configs["p_r"], real_configs["p_L"], real_configs["p_W"], pindexes)
    return (real_configs["p_n"],dis,dis*real_configs["p_n"])

def test_k():
    # k set from 1 to 10
    rr = 10
    l = []
    sys.argv.append("k:")
    for k in range(1, rr+1):
        sys.argv[-1] = "k:%s" % k
        l.append(main())
    # print for latex table
    for i, one in enumerate(l):
        if i == len(l)-1:
            print(r"%.1f\\" % one[1], end='')
        else:
            print("%.1f&" % one[1], end='')

if __name__ == "__main__":
    # test_k()
    print("#n/avgDist/totalDist:", main())
