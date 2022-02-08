#Python Noise addition class
import numpy as np
import math
import random


def addNoise(freqmat,domains,noise,nodecount,noisetype):
    """adds noise
    Args:
       freqmat:
       domains:
       ...
       noisetype:
    Returns:
    """
    if noise == 0.0:
       return freqmat,domains 
    if noisetype == "matrix":
       freqmat = addMatrixNoise(freqmat,noise)
    elif noisetype == "domain":   
       domains = addDomainNoise(domains,noise,nodecount)
    return freqmat,domains
    

def addMatrixNoise(freqmat,noiseparam):
    """adds noise to freqmat
    Args:
       freqmat:
       noiseparam:
    Returns:
       modfreqmat:
    """
    std = np.sqrt(np.amax(freqmat))*noiseparam
    modfreqmat = np.array(freqmat)
    for ind1 in xrange(np.shape(freqmat)[0]):
        for ind2 in xrange(np.shape(freqmat)[1]):
            modfreqmat[ind1,ind2] += int(round(random.gauss(0.0,std)))
            modfreqmat[ind1,ind2] = max(modfreqmat[ind1,ind2],0)
    return modfreqmat


def addDomainNoise(domains,noiseparam,nodecount):
    """adds noise to domains
    Args:
       domains:
       noiseparam:
       nodecount:
    Returns:
       moddomains:
    """
    std = noiseparam*math.sqrt(nodecount)
    moddomains = []
    for s,e in domains:
        s += int(round(random.gauss(0.0,std)))
        e += int(round(random.gauss(0.0,std)))
        if s>e:
           continue
        s = min(max(1,s),nodecount)
        e = min(max(1,e),nodecount)
        moddomains.append((s,e))
    return moddomains
