from random import *
import math
import itertools
import pickle
import numpy as np
import re

"""
List functions
"""
def uniques(li):
    return list(set(li))

#:[[a]] -> [a]
#http://stackoverflow.com/questions/716477/join-list-of-lists-in-python
def concat(lli, s='l'):
    if s=='s' or (len(lli)>0 and type(lli[0]) is str):
        return("".join(lli))
    else:
        return list(itertools.chain.from_iterable(lli))
#http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

def lines(s):
    return s.split('\n')

def unzip(li):
    return ([x for (x,y) in li], [y for (x,y) in li])

def mapkw(f, li):
    return [f(*l) for l in li]

def deepmap(f,li):
    if isinstance(li, list):
        return [deepmap(f, x) for x in li]
    else:
        return f(li)

#:[a] -> ([a],[a])
def split_eo(li):
    li1 = [li[i] for i in np.arange(0,len(li),2)]
    li2 = [li[i] for i in np.arange(1,len(li),2)]
    return li1, li2

#:[a] -> [[a]], segments of length length
def breakUpIntoSegments(li, length):
    li2 = []
    for i in range(len(li)-length+1):
        li2.append(li[i:i+length])
    return li2

def sample_without_replacement(n, k, method="take"):
    if method=="take":
        li = []
        i = 0
        for i in range(k):
            r = randrange(0,n)
            while r in li:
                r = randrange(0,n)
            li.append(r)
        return li
    else:
        li = []
        chosen = 0
        for i in range(n):
            r = random()
            if r < float(k-chosen)/(n-i):
                li.append(i)
                chosen = chosen + 1
            if chosen == k:
                break
        return li

def get_frac(li, i):
    j = math.floor(i)
    if i==j:
        return li[j]
    else:
        return (1-(i-j)) * li[j] + (i-j) * li[j+1]

def percentiles(li, ps):
    lis = sorted(li)
    l = len(li)
    return [get_frac(lis, i*(l-1)) for i in ps]

def quartiles(li):
    return percentiles(li, [0,.25, .5, .75, 1])

def e_(i,n):
    return [1 if j==i else 0 for j in range(n)]

"""
Strings
"""

def remove_chars(cs, s):
    rx = '[' + re.escape(cs) + ']'
    return re.sub(rx, '', s)

#np.asarray(

"""
Loops
"""
def nested_for1(curli, li, f):
    if len(li)==0:
        return f(curli)
    result = []
    for i in li[0]:
        curli2 = list(curli)
        curli2.append(i)
        result.append(nested_for1(curli2, li[1:], f))
    return result
        
def nested_for(li, f):
    return nested_for1([], li, f)

"""
def nested_for1_(curli, li, f):
    if len(li)==0:
        return f(curli)
    result = []
    for i in li[0]:
        curli2 = list(curli)
        curli2.append(i)
        
def nested_for_(li, f):
    return nested_for1_([], li, f)
"""

#Example
#nested_for([range(3), range(1,5)], lambda li: [li[0],li[0]**li[1]])
"""
[[[0, 0], [0, 0], [0, 0], [0, 0]],
 [[1, 1], [1, 1], [1, 1], [1, 1]],
 [[2, 2], [2, 4], [2, 8], [2, 16]]]
 """

def list_prod(llis):
    if llis==[]:
        return [[]]
    return [li+[x] for li in list_prod(llis[:-1]) for x in llis[-1]]
    #return [[x]+li for li in list_prod(llis[1:]) for x in llis[0]]

# like nested_for, but flattened.
def fors(llis, f):
    return [f(x) for x in list_prod(llis)]

def fors_(llis, f):
    for x in list_prod(llis):
        f(x)

def fors_zip(llis, f):
    return fors(llis, lambda x: (x, f(x)))

"""
Dictionaries
"""
def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

def map_keys(f, d):
    return {f(k): v for (k,v) in d.items()}

def map_vals(f, d):
    return {k : f(v) for (k,v) in d.items()}

def get_with_default(d, k, default):
    return d[k] if k in d else default

"""
Print functions
"""
def printv(s, v, lim):
    if v >= lim:
        print(s)

"""
Other
"""

def pickle_load(fname):
    with open(fname, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
        data = pickle.load(f)
    f.close()
    return data

def pickle_dump(fname, data):
    with open(fname, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()
