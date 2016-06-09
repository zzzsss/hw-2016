import random
import operator
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# generate the random init points
def random_init(n, L, W):
    # init n points (x,y) randomly within (L, W)
    # return a list of 2-elem-list
    ret = []
    for i in range(n):
        x = L * random.random()
        y = W * random.random()
        ret.append([x, y])
    ret.sort(key=(lambda z: z[0]))
    return ret


# splitter for k-barrier
def split_k(plist, k, valuer):
    # return the list (len==k) of list of indexes for the split
    # selector should be: int selector(plist, remain_indexes_set, this_index_list);
    length = len(plist)
    indexes = [[] for i in range(k)]
    remains = set([i for i in range(length)])
    turn = 0
    while len(remains) > 0:
        # value each point independently
        value_dict = {}
        for i in remains:
            value_dict[i] = valuer([plist[z] for z in indexes[turn]], plist[i])
        # add one
        this_one = min(value_dict.items(), key=operator.itemgetter(1))[0]
        indexes[turn].append(this_one)
        remains.remove(this_one)
        turn = (turn+1) % k
    for one in indexes:
        one.sort()
    return indexes


# valuer
def _valuer_x(all, one, r):
    s = 0
    for p in all:
        if abs(one[0]-p[0]) <= r:
            s += abs(one[0]-p[0])
    return s

def value_x(r):
    return lambda all, one: _valuer_x(all, one, r)

def valuer_random(all, one):
    return random.random()

def valuer_y(all, one):
    s = 0
    for p in all:
        s += abs(one[1]-p[1])
    return s

def _valuer_xy(all, one, r, weight):
    # consider both: x-cover + weight * y-distance
    return value_x(r)(all, one) + weight * valuer_y(all, one)

def valuer_xy(r, weight):
    return lambda all, one: _valuer_xy(all, one, r, weight)

COLORS = ['b', 'r', 'g', 'y', 'k', 'm', 'c']
# draw one
def draw_one(before, after, r, L, W, pindexes):
    t = "".join(time.ctime().split()[3].split(":"))     #tag
    r *= 2
    # plot it
    fig1 = plt.figure(t+"before", figsize=(10, 10))
    ax = plt.gca()
    ax.set_xticks(np.linspace(0,L,11))
    ax.set_yticks(np.linspace(-L/2,L/2,11))
    for i, x in enumerate(before):
        plt.annotate("p%s" % i, xy=(x[0], x[1]))
        circle = Ellipse((x[0], x[1]), r, r, color='r', fill=False)
        ax.add_patch(circle)
    # plt.show()
    # plt.savefig(t+"before")
    # plt.cla()
    fig2 = plt.figure(t+"after", figsize=(10, 10))
    ax = plt.gca()
    ax.set_xticks(np.linspace(0,L,11))
    ax.set_yticks(np.linspace(-L/2,L/2,11))
    for i, x in enumerate(after):
        plt.annotate("p%s" % i, xy=(x[0], x[1]))
        group_i = 0
        for ind, group in enumerate(pindexes):
            if i in group:
                group_i = ind
                break
        circle = Ellipse((x[0], x[1]), r, r, color=COLORS[group_i % len(COLORS)], fill=False)
        ax.add_patch(circle)
    plt.show()
    # plt.savefig(t+"after")
    # plt.cla()

def test_draw():
    draw_one(random_init(20, 100, 10), random_init(20, 100, 10), 5, 100, 10)
