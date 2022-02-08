#Histone Evaluation utilities
import os
import sys
import math
import itertools
import random
import numpy as np
from munkres import Munkres
from copy import deepcopy
import sklearn.metrics

def testVicutEstimate(alldoms,nodecount,negdict):
    """tests score estimation
    """          
    for tind in xrange(1000):
        print "iter"
        k = 50
        seldoms = []
        mydoms = list(alldoms)
        random.shuffle(mydoms)
        tval = 0.0
        for start,end in mydoms:
            flag = True
            for start1,end1 in seldoms:
                interlen = max(0, min(end, end1) - max(start, start1)+1)
                if interlen > 0:
                   flag = False
                   break
            if flag:
               seldoms.append((start,end))
            if len(seldoms) >= k:
               break
        empones = sorted([(block[0],block[-1]) for block in addEmptyClusters(seldoms,range(1,nodecount+1))])
        for start,end in seldoms:
            empones.remove((start,end))
        for start,end in empones:
            assert negdict.has_key((start,end))

def estViCutCoefsExtend(alldoms,inferdoms,empdoms,rempart,nodecount):
    """est vicut coefs
    """
    maxneglen = max([end-start+1 for start,end in inferdoms])
    posdict,negdict = {},{}
    for start,end in alldoms:
        curweight = 0.0
        for start1,end1 in inferdoms:
            interlen = max(0, min(end, end1) - max(start, start1)+1)
            if interlen!=0:
               curweight -= interlen*(math.log(float(interlen)/(end-start+1)) + math.log(float(interlen)/(end1-start1+1)))
        interlen = 0       
        for start1,end1,sym1 in rempart:
            if sym1 == "e":
               continue 
            interlen += max(0, min(end, end1) - max(start, start1)+1)
        if interlen!=0:
           curweight -= interlen*(math.log(float(interlen)/(end-start+1)) + math.log(float(interlen)/(nodecount)))
        #interlen = 0       
        #for start1,end1 in empdoms:
        #    interlen += max(0, min(end, end1) - max(start, start1)+1)
        #interlen2 = 0      
        #for start1,end1,sym1 in rempart:
        #    if sym1 == "d":
        #       continue 
        #    interlen2 += max(0, min(end, end1) - max(start, start1)+1)
        #interlen += interlen2        
        #if interlen!=0:
        #   curweight -= interlen*(math.log(float(interlen)/(end-start+1)) + math.log(float(interlen)/(nodecount)))
        assert curweight >= 0.0
        posdict[(start,end)] = curweight
    uselefts = [(item,"l") for item in sorted(list(set(item[0] for item in alldoms) | set([1])))]
    userights = [(item,"r") for item in sorted(list(set(item[1] for item in alldoms) | set([nodecount])))]
    for (start,sym1),(end,sym2) in itertools.combinations(sorted(uselefts+userights),2):
        ustart = start if sym1 == "l" else start+1
        uend = end-1 if sym2 == "l" else end
        if end-start > 250*maxneglen: #if sym1 == "r" and sym2 == "l" and end-start > 200:
           continue
        curweight = 0.0
        for start1,end1 in empdoms:
            interlen = max(0, min(uend, end1) - max(ustart, start1)+1)
            if interlen!=0:
               curweight -= interlen*(math.log(float(interlen)/(uend-ustart+1)) + math.log(float(interlen)/(end1-start1+1)))
        interlen = 0       
        for start1,end1,sym1 in rempart:
            if sym1 == "d":
               continue 
            interlen += max(0, min(uend, end1) - max(ustart, start1)+1)
        if interlen!=0:
           curweight -= interlen*(math.log(float(interlen)/(uend-ustart+1)) + math.log(float(interlen)/(nodecount)))
        #interlen = 0       
        #for start1,end1 in inferdoms:
        #    interlen = max(0, min(uend, end1) - max(ustart, start1)+1)
        #    if interlen!=0:
        #       curweight -= interlen*(math.log(float(interlen)/(nodecount)) + math.log(float(interlen)/(end1-start1+1)))
        #interlen = 0       
        #for start1,end1,sym1 in rempart:
        #    if sym1 == "e":
        #       continue 
        #    interlen += max(0, min(uend, end1) - max(ustart, start1)+1)       
        #if interlen!=0:
        #   curweight -= interlen*(math.log(float(interlen)/(nodecount)) + math.log(float(interlen)/(nodecount)))
    
        assert curweight >= 0.0
        if negdict.has_key((ustart,uend)):
           assert negdict[(ustart,uend)] == curweight 
        if not negdict.has_key((ustart,uend)):
           negdict[(ustart,uend)] = curweight     
    return posdict,negdict

         
def estViCutCoefs(alldoms,usedoms,inferdoms,nodecount):
    """est vicut coefs
    Args:
       alldoms,usedoms,inferdoms,nodecount:
    Returns:
    """
    maxneglen = max([end-start+1 for start,end in usedoms if (start,end) not in inferdoms])
    posdict,negdict = {},{}
    for start,end in alldoms:
        curweight = 0.0
        for start1,end1 in usedoms:
            interlen = max(0, min(end, end1) - max(start, start1)+1)
            if interlen!=0:
               curweight -= interlen*(math.log(float(interlen)/(end-start+1)) + math.log(float(interlen)/(end1-start1+1)))
        assert curweight >= 0.0
        posdict[(start,end)] = curweight
    uselefts = [(item,"l") for item in sorted(list(set(item[0] for item in alldoms) | set([1])))]
    userights = [(item,"r") for item in sorted(list(set(item[1] for item in alldoms) | set([nodecount])))]
    for (start,sym1),(end,sym2) in itertools.combinations(sorted(uselefts+userights),2):
        ustart = start if sym1 == "l" else start+1
        uend = end-1 if sym2 == "l" else end
        if end-start > 3*maxneglen: #if sym1 == "r" and sym2 == "l" and end-start > 200:
           continue
        curweight = 0.0
        for start1,end1 in usedoms:
            interlen = max(0, min(uend, end1) - max(ustart, start1)+1)
            if interlen!=0:
               curweight -= interlen*(math.log(float(interlen)/(uend-ustart+1)) + math.log(float(interlen)/(end1-start1+1)))
        assert curweight >= 0.0
        if not negdict.has_key((ustart,uend)):
           negdict[(ustart,uend)] = curweight
    return posdict,negdict
        
def estViCutScore(nodecount,alldoms,inferdoms,vitype):
    """estimates vi cut score
    Args:
       nodecount:
       alldoms : variable
       inferdoms: fixed
       vitype: extend,normal
    Returns:
       viscore:   
    """    
    assert len(alldoms) == len(set(alldoms)) and vitype in ["extend","normal"]
    if vitype == "extend":
       empdoms = [(block[0],block[-1]) for block in addEmptyClusters(inferdoms,range(1,nodecount+1)) if (block[0],block[-1]) not in inferdoms]
       rempart = [(start,end,"e") for start,end in inferdoms] + [(start,end,"d") for start,end in empdoms]
       posdict,negdict = estViCutCoefsExtend(alldoms,inferdoms,empdoms,rempart,nodecount)
    elif vitype == "normal":         
       usedoms = sorted([(block[0],block[-1]) for block in addEmptyClusters(inferdoms,range(1,nodecount+1))])
       posdict,negdict = estViCutCoefs(alldoms,usedoms,inferdoms,nodecount)
    #testVicutEstimate(alldoms,nodecount,negdict)   
    MAXVAL = 1000000000000000000.0
    posobjvals,negobjvals = [MAXVAL] * (nodecount+1), [MAXVAL] * (nodecount+1)
    posoptsols,negoptsols = {}, {}
    if posdict.has_key((1,2)):
       posobjvals[2] = posdict[(1,2)]
       posoptsols[2] = [(1,2,"d")]
    for tind in [1,2]:   
        if negdict.has_key((1,tind)):
           negobjvals[tind] = negdict[(1,tind)]
           negoptsols[tind] = [(1,tind,"e")]
    for end in xrange(3,nodecount+1):
        minval,optsol = MAXVAL,None
        if posdict.has_key((1,end)):
           if posdict[(1,end)] < minval:
              minval = posdict[(1,end)]
              optsol = [(1,end,"d")]
        for k1 in xrange(2,end-1):
            if posoptsols.has_key(k1) and posdict.has_key((k1+1,end)):
               curval = posobjvals[k1] + posdict[(k1+1,end)] 
               if curval < minval:
                  minval = curval
                  optsol = posoptsols[k1] + [(k1+1,end,"d")]
            if negoptsols.has_key(k1) and posdict.has_key((k1+1,end)):      
               curval = negobjvals[k1] + posdict[(k1+1,end)] 
               if curval < minval:
                  minval = curval
                  optsol = negoptsols[k1] + [(k1+1,end,"d")]
        if minval != MAXVAL:
           posobjvals[end] = minval
           posoptsols[end] = list(optsol)
                     
        minval,optsol = MAXVAL,None  
        if negdict.has_key((1,end)):
           if negdict[(1,end)] < minval:
              minval = negdict[(1,end)]
              optsol = [(1,end,"e")]
        for k1 in xrange(2,end):
            if posoptsols.has_key(k1) and negdict.has_key((k1+1,end)):
               curval = posobjvals[k1] + negdict[(k1+1,end)]
               if curval < minval:
                  minval = curval
                  optsol = posoptsols[k1] + [(k1+1,end,"e")]
        if minval != MAXVAL:
           negobjvals[end] = minval
           negoptsols[end] = list(optsol)
    maxnode = None
    for tnode in xrange(1,nodecount+1):
        if negoptsols.has_key(tnode):
           maxnode = tnode
    print maxnode
    print len(alldoms)
    print "done done"
    #exit(1)
    for tind,(s1,e1,sym1) in enumerate(negoptsols[nodecount][0:-1]):
        if sym1 == "e":
           assert negoptsols[nodecount][tind+1][2] == "d"
    negoptsol = [(start,end) for start,end,sym in negoptsols[nodecount] if sym=="d"]
    #print "coverage: ",sum([end-start+1 for start,end in negoptsol])
    negvi = getVIScoreExtend(negoptsol,inferdoms,nodecount)
    posvi = 100000.0
    if posoptsols.has_key((1,nodecount)):
       posoptsol = [(start,end) for start,end,sym in posoptsols[nodecount] if sym=="d"]
       posvi = getVIScore(posoptsol,inferdoms,nodecount)
    return min(posvi,negvi)


def getMI(part1,part2,nodecount):
    """returns mi score
    Args:
       part1:
       part2:
       nodecount:
    Returns:
       miscore:
    """
    pdist1 = {index: float(len(part1[index]))/nodecount for index in xrange(len(part1))}
    pdist2 = {index: float(len(part2[index]))/nodecount for index in xrange(len(part2))}
    miscore = 0.0
    for ind1 in xrange(len(part1)):
         for ind2 in xrange(len(part2)):
             p = float(len(set(part1[ind1]).intersection(set(part2[ind2]))))/nodecount
             if p != 0:
                miscore += p*(math.log(p/(pdist1[ind1]*pdist2[ind2]),2.0))
    return miscore

def getVI(part1,part2,nodecount):
    """returns vi score
    Args:
       part1:
       part2:
       nodecount:
    Returns:
       miscore:
    """
    pdist1 = {index: float(len(part1[index]))/nodecount for index in xrange(len(part1))}
    pdist2 = {index: float(len(part2[index]))/nodecount for index in xrange(len(part2))}
    viscore = getEntropy(pdist1) + getEntropy(pdist2) - 2.0*getMI(part1,part2,nodecount)
    if abs(viscore) < 0.001:
       return 0.0
    return viscore
    
def getEntropy(probdist):
    """gets entropy
    Args:
       probdist:
    Returns:
       entsum:
    """
    entsum = sum([-1.0*probdist[item]*math.log(probdist[item],2.0) for item in probdist.keys() if probdist[item] != 0])
    return entsum

def addEmptyClusters(domains,allnodes):
    """adds empty clusters
    Args:
       domains,allnodes:
    """        
    part1 = sorted([range(start,end+1) for start,end in domains], key=lambda item: item[0])
    curin,part2 = 1,[]
    for clust in part1:
        start,end = min(clust),max(clust)
        if curin <= start:
           part2.append(range(curin,start))
        curin = end+1
    if curin <= len(allnodes):
       part2.append(range(curin,len(allnodes)+1))
    if True:   
       for clust1 in part1:
           for clust2 in part2:
               assert len(set(clust1) & set(clust2)) == 0
    part1.extend(part2)
    part1 = [part for part in part1 if len(part)>0]
    if True:  
       for ind1 in xrange(len(part1)):
           range1 = set(part1[ind1]) 
           for ind2 in xrange(ind1+1,len(part1)):
               range2 = set(part1[ind2])
               assert len(range1 & range2) == 0
       assert len(set(node for part in part1 for node in part) ^ set(allnodes)) == 0        
    return part1


def matchPenalized(scoremat,fixedval,penalty,truelen,estlen):
    """matches also with mismatch penalty
    Args:
       scoremat,fixedval,penalty:
       truelen,estlen:
    Returns:
       curscores:   
    """
    m = Munkres()
    tscoremat = deepcopy(scoremat)  
    indexes = m.compute(tscoremat)
    print "det match info"
    for ind1,ind2 in indexes:
        if ind1 in [1, 6]:
           print ind1,ind2,scoremat[ind1,ind2],min(scoremat[ind1,:]) 
        if ind2 in [4, 7]:
           print ind1,ind2,scoremat[ind1,ind2],min(scoremat[:,ind2])     
    print "lens: ",estlen,truelen
    fixcount = 0
    for ind1 in xrange(np.shape(scoremat)[0]):
        for ind2 in xrange(np.shape(scoremat)[1]):
            if abs(scoremat[ind1,ind2]) <= 0.01:
               fixcount += 1   
    print "fixcount: ",fixcount                    
    zercount = 0        
    for ind1 in xrange(np.shape(scoremat)[0]):
        for ind2 in xrange(np.shape(scoremat)[1]):
            if abs(scoremat[ind1,ind2]) <= 0.01:
               zercount += 1    
    print "prematch zercount: ",zercount
    curscores = [scoremat[ind1,ind2] for ind1,ind2 in indexes if scoremat[ind1,ind2] != fixedval]
    #zervals = [item for item in curscores if abs(item) < 0.01]
    zervals = [(ind1,ind2) for ind1,ind2 in indexes if scoremat[ind1,ind2] < 0.01]
    print "inside penal: ",zervals
    remindices = [(ind1,ind2) for ind1,ind2 in indexes if scoremat[ind1,ind2] == fixedval]      
    assert len(remindices) == abs(estlen - truelen)
    if truelen > estlen: 
       curscores.extend([penalty * min(scoremat[tind,:]) for tind,eind in remindices])
    elif truelen < estlen: 
       curscores.extend([penalty * min(scoremat[:,eind]) for tind,eind in remindices])
    return curscores

def getScoreMat(truelist,estlist,fixedval):
    """
    """ 
    minlen,maxlen = min(len(estlist),len(truelist)), max(len(estlist),len(truelist))
    scoremat = np.full((maxlen, maxlen), fixedval)
    for trind,trbound in enumerate(truelist):
        for estind,estbound in enumerate(estlist):
            scoremat[trind,estind] = abs(trbound - estbound)**0.5  
    zercount = 0
    print "working"        
    for ind1 in xrange(np.shape(scoremat)[0]):
        for ind2 in xrange(np.shape(scoremat)[1]):
            if abs(scoremat[ind1,ind2]) <= 0.01:
               print truelist[ind1], estlist[ind2], ind1, ind2
               zercount += 1
    print "zercount: ",zercount
    return scoremat

def estBoundMatchScore(truebounds,estbounds,penalty=2.0):
    """estimates match-based score over boundaries
    Args:
       truebounds,estbounds: dictionary of start and end positions
       penalty: fixedcost per missed domain's mismatch
    Returns: 
       score:   
    """ 
    fixedval = 100000000000
    scoremat1, scoremat2 = getScoreMat(truebounds["start"], estbounds["start"],fixedval), getScoreMat(truebounds["end"], estbounds["end"],fixedval) 
    curscores = []
    for cind,cscoremat in enumerate([scoremat1,scoremat2]):
        truelen, estlen = (len(truebounds["start"]), len(estbounds["start"])) if cind == 0 else (len(truebounds["end"]), len(estbounds["end"])) 
        tcumscores = matchPenalized(cscoremat,fixedval,penalty,truelen,estlen)   
        zercount = sum(1 for item in tcumscores if abs(item) < 0.001)
        print "inside: ",zercount
        curscores.extend(tcumscores) 
    cumscore = sum(curscores)    
    len2dist = {}
    for score in curscores:
        len2dist.setdefault(score,0)
        len2dist[score] += 1
    print len2dist 
    print "mean: ",np.mean(curscores)
    print "std: ",np.std(curscores)  
    exit(1)    
    return np.mean(curscores)


def estPointMatchScore(truebounds,estbounds,penalty=2.0):
    """estimates match-based score over boundaries independently
    Args:
       truebounds,estbounds: start and end positions
       penalty: fixedcost per missed domain's mismatch
    Returns: 
       score:  
    """ 
    fixedval = 100000000000
    scoremat = getScoreMat(truebounds, estbounds, fixedval)
    truelen, estlen = len(truebounds), len(estbounds)
    curscores = matchPenalized(scoremat,fixedval,penalty,truelen,estlen)
    #return np.mean(curscores)
    

    scoremat = np.zeros((len(truebounds),len(estbounds)),dtype=np.float64)
    for trind,trbound in enumerate(truebounds):
        for estind,estbound in enumerate(estbounds):
            scoremat[trind,estind] = abs(trbound - estbound)
    scores = []
    for trind in xrange(len(truebounds)):
        score = min(scoremat[trind,:])
        scores.append(score)
    print "infom:"    
    print "mean ",np.mean(scores)
    print "median ",np.median(scores)
    print "std ",np.std(scores)
    len2dist = {}
    for mylen in scores:
        len2dist.setdefault(mylen,0)
        len2dist[mylen] += 1  
    print len2dist
    print "done"
    len2dist[50] = 10
    for ind in xrange(8):
        scores.append(50)
    print np.mean(scores)    
    #len2dist = {0.0: 10, 1.0: 14, 2.0: 7, 3.0: 3, 4.0: 8, 5.0: 5, 6.0: 7, 7.0: 2, 8.0: 6, 9.0: 5, 10.0: 2, 11.0: 5, 12.0: 2, 15.0: 1, 16.0: 1, 20.0: 3, 22.0: 3, 24.0: 1, 27.0: 1}
    exit(1)                         
    m = Munkres()
    scoremat2 = deepcopy(scoremat)  
    indexes = m.compute(scoremat2)
    curscore = sum(scoremat[ind1,ind2] for ind1,ind2 in indexes)
    print "curscore is: ",curscore
    exit(1)
    print curscore/float(min(len(truedoms),len(estdoms)))
    if len(truedoms) < len(estdoms):
       usedinds = set(eind for tind,eind in indexes)
       reminds = set(range(0,len(estdoms))).difference(usedinds)
       curscore += sum(min(scoremat[:,eind]) for eind in reminds) 
    return curscore/float(max(len(truedoms),len(estdoms)))


def estMatchScore(truedoms,estdoms):
    """estimates match-based score
    Args:
       truedoms,estdoms:
    Returns:   
    """ 
    scoremat = np.zeros((len(truedoms),len(estdoms)),dtype=np.float64)
    for trind,trdom in enumerate(truedoms):
        for estind,estdom in enumerate(estdoms):
            scoremat[trind,estind] = abs(trdom[0] - estdom[0]) + abs(trdom[1] - estdom[1])            
    m = Munkres()
    scoremat2 = deepcopy(scoremat)  
    indexes = m.compute(scoremat2)
    curscore = sum(scoremat[ind1,ind2] for ind1,ind2 in indexes)
    print len(truedoms),len(estdoms)
    print curscore
    print curscore/float(min(len(truedoms),len(estdoms)))
    if len(truedoms) < len(estdoms):
       usedinds = set(eind for tind,eind in indexes)
       reminds = set(range(0,len(estdoms))).difference(usedinds)
       curscore += sum(min(scoremat[:,eind]) for eind in reminds) 
    return curscore/float(max(len(truedoms),len(estdoms)))


def estPredScoreBlockList(truedoms,inferdoms):
    """estimates prediction score list blockwise
    Args:
       truedoms,inferdoms:
    Returns:
       scoredict:     
    """
    tp,fp,fn = 0,0,0
    for start1,end1 in inferdoms:
        estset = set(range(start1,end1+1))
        flag = False
        for start2,end2 in truedoms:
            trueset = set(range(start2,end2+1))
            interset = estset & trueset
            unionset = estset | trueset
            if len(interset) >= len(unionset)*0.5: 
               flag = True
               break
        if flag:   
           tp += 1
        else:
           fp += 1      
    #tp2=0       
    #for start1,end1 in truedoms:
    #    trueset = set(range(start1,end1+1))
    #    flag = False
    #    for start2,end2 in inferdoms:
    #        estset = set(range(start2,end2+1))
    #        interset = estset & trueset
    #        unionset = estset | trueset
    #        if len(interset) >= len(unionset)*0.5: 
    #           flag = True
    #           break
    #    if flag:   
    #       tp2 += 1
    #assert tp == tp2       
    fn = len(truedoms) + len(inferdoms) - tp - fp
    scoredict = {}
    scoredict["npc"] = float(tp)/(tp+fp+fn)
    scoredict["prec"] = 0.0
    if tp+fp != 0:
       scoredict["prec"] = float(tp)/(tp+fp)
    scoredict["sen"] = 1.0    
    if tp+fn != 0:   
       scoredict["sen"] = float(tp)/(tp+fn)
    scoredict["f1"] = (2.0*tp)/((2.0*tp)+fp+fn)

    tp,fp,fn = 0,0,0
    for start1,end1 in inferdoms:
        estset = set(range(start1,end1+1))
        flag = False
        for start2,end2 in truedoms:
            trueset = set(range(start2,end2+1))
            interset = estset & trueset
            if len(interset) >= len(estset)*0.8: 
               flag = True
               break
        if flag:   
           tp += 1
        else:
           fp += 1
    fn = len(truedoms) + len(inferdoms) - tp - fp
    scoredict = {}
    scoredict["npc"] = float(tp)/(tp+fp+fn)
    scoredict["prec"] = 0.0
    if tp+fp != 0:
       scoredict["prec"] = float(tp)/(tp+fp)
    scoredict["sen"] = 1.0    
    if tp+fn != 0:   
       scoredict["sen"] = float(tp)/(tp+fn)
    scoredict["f1"] = (2.0*tp)/((2.0*tp)+fp+fn)
    return scoredict


def getScore(truedoms,inferdoms,nodecount,scoretype):
    """gets score
    Args:
       truedoms,inferdoms:
       nodecount,scoretype:
    Returns:   
    """
    if scoretype in ["npc","acc","prec","sen","fpr","f1","mcc","jaccard"]:
       scoredict = estPredScoreList(truedoms,inferdoms,nodecount)
       return scoredict[scoretype]
    elif scoretype == "nvi":
       return getVIScore(truedoms,inferdoms,nodecount)

def getVIScoreExtend(truedoms,inferdoms,nodecount):
    """gets VI score extended
    Args:
       truedoms,inferdoms,nodecount:
    Returns:
       viscore:
    """
    allnodes = range(1,nodecount+1)
    def getParts(usedoms):
        """estimates modified parts
        """
        empusedoms = [(block[0],block[-1]) for block in addEmptyClusters(usedoms,allnodes) if (block[0],block[-1]) not in usedoms]
        sentpart,rempart = [], []
        for start,end in usedoms:
            curpart = range(start,end+1) 
            sentpart.append(curpart)
            rempart.extend(range(nodecount+start,nodecount+end+1))
        for start,end in empusedoms:
            curpart = range(nodecount+start,nodecount+end+1) 
            sentpart.append(curpart)
            rempart.extend(range(start,end+1))    
        sentpart.append(rempart)
        return sentpart 
    truepart = getParts(truedoms)
    estpart = getParts(inferdoms)
    for trialpart in [truepart,estpart]:
        for block1,block2 in itertools.combinations(trialpart,2):
            assert max(0, min(block1[-1], block2[-1]) - max(block1[0], block2[0])+1) == 0
        seennodes = [item for block in trialpart for item in block]
        assert len(seennodes) == 2*nodecount and len(seennodes) == len(set(seennodes)) and max(seennodes) == 2*nodecount        
    viscore = getVI(truepart,estpart,2*nodecount)
    viscore /= float(math.log(2*nodecount,2.0))
    return viscore

def getVIScore(truedoms,inferdoms,nodecount):
    """gets VI score
    Args:
       truedoms,inferdoms,nodecount:
    Returns:
    """
    allnodes = range(1,nodecount+1) 
    truepart = addEmptyClusters(truedoms,allnodes)
    estpart = addEmptyClusters(inferdoms,allnodes)
    viscore = getVI(truepart,estpart,nodecount)
    viscore /= float(math.log(nodecount,2.0))
    return viscore


def estPredScoreList(truedoms,inferdoms,nodecount):
    """estimates prediction score list
    Args:
       truedoms,inferdoms:
       nodecount,scoretype:
    Returns:
       scoredict:     
    """    
    truevec,infervec = getVec(truedoms,nodecount), getVec(inferdoms,nodecount)
    [[tn, fp], [fn, tp]] = sklearn.metrics.confusion_matrix(truevec, infervec)

    if False:
     print "tp: ",tp
     print "fp: ",fp
     print "fn: ",fn
     print "tn: ",tn
     fpset,fnset = set(), set()
     for tind,item1 in enumerate(truevec): 
        item2 = infervec[tind]
        if item1 == 1 and item2 == 0:
           fnset.add(tind)
        elif item1 == 0 and item2 == 1:
           fpset.add(tind)
     print "fpset: ",fpset
     print "fnset: ",fnset
        
    assert nodecount == tn + fn + fp + tp
    scoredict = {}
    scoredict["jaccard"] = float(tp)/(tp+fp+fn)
    scoredict["acc"] = float(tp+tn)/nodecount
    scoredict["prec"] = 0.0
    if tp+fp != 0:
       scoredict["prec"] = float(tp)/(tp+fp)
    scoredict["sen"] = 1.0    
    if tp+fn != 0:   
       scoredict["sen"] = float(tp)/(tp+fn)
    scoredict["fpr"] = float(fp)/(fp+tn)
    scoredict["f1"] = (2.0*tp)/((2.0*tp)+fp+fn)
    if (tp+fp) == 0.0 or (tp+fn) == 0.0 or (tn+fp) == 0.0 or (tn+fn) == 0.0:
       scoredict["mcc"] = 0.0
    else:  
       scoredict["mcc"] = float((tp*tn)-(fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))     
    return scoredict

def getVec(doms,nodecount):
    """get vector
    Args:
       doms,nodecount:
    Returns:   
    """
    sentvec = np.zeros(nodecount,dtype=np.int)
    for start,end in doms:
        sentvec[start-1:end] = 1
    return sentvec


def estAvgDistance(truedoms,inferdoms):
    """est avg distance
    Args:
       truedoms,inferdoms:
    Returns:
       meandist,meandist2:   
    """
    truelocs,estlocs = [item for items in truedoms for item in items], [item for items in inferdoms for item in items]
    avgdist,avgdist2 = 0.0, 0.0
    if len(estlocs)!= 0:
       away = 0
       for loc in estlocs:
           mindist = min([abs(trloc-loc) for trloc in truelocs])
           if mindist > 1:
              away += 1
       away /= float(len(estlocs))       
       for loc in estlocs:
           mindist = min([abs(trloc-loc) for trloc in truelocs])
           avgdist += mindist
       meandist = avgdist/len(estlocs)
       avgdist2 = avgdist
       for loc in truelocs:
           mindist = min([abs(estloc-loc) for estloc in estlocs])
           avgdist2 += mindist 
       meandist2 = avgdist2/float(len(estlocs)+len(truelocs))
    else:
       meandist,meandist2 = 10000.0, 10000.0   
    return meandist,meandist2,away
