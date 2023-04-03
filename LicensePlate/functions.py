import numpy as np

def distance(x,y):
    (x1,y1,_,_) = x
    (x2,y2,_,_) = y
    dx = x2 - x1
    dy = y2 - y1
    return dx*dx+dy*dy

def findCluster(letterLike):
    tracked = {}
    for i,_ in enumerate(letterLike):
        tracked[i] = False
    cluster = []
    i=0
    while (i<len(letterLike)):
        if not tracked[i]:
            cluster.append([])
            c = cluster[-1]
            c.append(i)
            tracked[i] = True
            i+=1
            j = 0
            while(j<len(c)):
                for k,_ in enumerate(letterLike):
                    if not tracked[k]:
                        if distance(letterLike[c[j]],letterLike[k]) < 500:
                            tracked[k] = True
                            c.append(k)
                j+=1
        else:
            i += 1
    if len(cluster) == 0:
        numberPlateCluster = []
    elif len(cluster) == 1:
        numberPlateCluster = cluster[0]
    else:
        numberPlateCluster = cluster[np.argmax([len(i) for i in cluster])]
    numbers = []
    # print(numberPlateCluster)
    if len(numberPlateCluster)>=6:
        for c in numberPlateCluster:
            numbers.append(letterLike[c])
    return numbers