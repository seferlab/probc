#Normalized by ICE by Imaakev 2012
import numpy as np
import math

def normMatrixICE(freqmat,MAXCOUNT=250):
    """normalize matrix ICE by Imaakev 2012 new version
    Args:
       freqmat:
    Returns:
       normmat:
    """
    fqsum = np.sum(freqmat)
    normmat,matsize = np.array(freqmat,dtype=np.float), np.shape(freqmat)[0]
    zerolocs = set(ind for ind in xrange(matsize) if np.sum(normmat[ind,:]) < 0.0001)
    remlocs = sorted(list(set(range(matsize)) - zerolocs))
    normmat = normmat[np.ix_(remlocs,remlocs)]
    savemat = np.array(normmat)
    matsize = np.shape(normmat)[0]
    B = np.array([1.0]* matsize)
    predeltaB = None
    for itcount in xrange(MAXCOUNT):
        Svec = normmat.sum(axis=1)
        Smean = np.mean(Svec)
        deltaB = Svec/Smean
        assert abs(np.mean(deltaB) -1.0) < 0.001
        normmat /= np.dot(deltaB.reshape(matsize,1),deltaB.reshape(1,matsize))
        B *= deltaB
        print itcount,sum(np.abs(deltaB)), np.sum(normmat), np.sum(B), sum(np.abs(deltaB-predeltaB)) if predeltaB != None else None
        if predeltaB!=None and sum(np.abs(deltaB-predeltaB)) < 0.0001:
           break 
        predeltaB = np.array(deltaB)
    tcoef = fqsum / np.sum(normmat)
    normmat *= tcoef
    newmatsize = np.shape(freqmat)[0]
    newnormmat = np.zeros((newmatsize,newmatsize),dtype=np.float)
    for ind1 in xrange(matsize):
        for ind2 in xrange(matsize):
            newnormmat[remlocs[ind1],remlocs[ind2]] = normmat[ind1,ind2]
    normmat = np.around(newnormmat, decimals=5)
    return normmat


def NOTUSED_ICE():
    """
    """
    if True:
       matsize = np.shape(normmat)[0]
       B = np.array([1.0]* matsize)
       predeltaB,itercount = None,0  
       while True:
          itercount += 1 
          Svec = normmat.sum(axis=1)
          Smean = np.mean(Svec[np.ix_(remlocs)])
          deltaB = Svec/Smean
          for zeroloc in zerolocs:
              deltaB[zeroloc] = 1.0
          for item in deltaB:
              assert item >= 0.0000001    
          normmat /= np.dot(deltaB.reshape(matsize,1),deltaB.reshape(1,matsize))
          B *= deltaB
          print "imp ",sum(np.abs(deltaB)), np.sum(normmat), np.sum(B), sum(np.abs(deltaB-predeltaB)) if predeltaB != None else None
          if (predeltaB!=None and sum(np.abs(deltaB-predeltaB)) < 0.0001) or itercount >= 200:
            break 
          predeltaB = np.array(deltaB)
       tcoef = fqsum / np.sum(normmat)
       normmat *= tcoef

       #indices = [(node,node) for node in xrange(matsize)]
       #indices.extend([(node,node+1) for node in xrange(matsize-1)])
       #indices.extend([(node+1,node) for node in xrange(matsize-1)])
       #assert len(indices) == len(set(indices))
       #indsum = sum([normmat[start,end] for start,end in indices])
       #remsum = np.sum(normmat) - indsum
       #tcoef = fqsum / float(remsum)
       #normmat *= tcoef
       #for start,end in indices:
       #    normmat[start,end] /= tcoef
         
       #remsum = np.sum(normmat) - sum([normmat[node,node] for node in xrange(matsize)])
       #tcoef = fqsum / float(remsum)
       #normmat *= tcoef
       #for node in xrange(matsize):
       #    normmat[node,node] /= tcoef
       normmat = np.around(normmat, decimals=5)
       return normmat
    else:     
       normmat = normmat[np.ix_(remlocs,remlocs)]
       savemat = np.array(normmat)
       matsize = np.shape(normmat)[0]
       B = np.array([1.0]* matsize)
       predeltaB,itercount = None,0
       while True:
          itercount += 1 
          Svec = normmat.sum(axis=1)
          Smean = np.mean(Svec)
          deltaB = Svec/Smean
          assert abs(np.mean(deltaB) -1.0) < 0.001
          normmat /= np.dot(deltaB.reshape(matsize,1),deltaB.reshape(1,matsize))
          B *= deltaB
          print "imp ",sum(np.abs(deltaB)), np.sum(normmat), np.sum(B)
          if (predeltaB!=None and sum(np.abs(deltaB-predeltaB)) < 0.0001) or itercount >= 200:
            break 
          predeltaB = np.array(deltaB)
       tcoef = fqsum / np.sum(normmat)
       normmat *= tcoef
       newmatsize = np.shape(freqmat)[0]
       newnormmat = np.zeros((newmatsize,newmatsize),dtype=np.float)
       for ind1 in xrange(matsize):
           for ind2 in xrange(matsize):
               newnormmat[remlocs[ind1],remlocs[ind2]] = normmat[ind1,ind2]
       normmat = np.around(newnormmat, decimals=5)
       return normmat
      
