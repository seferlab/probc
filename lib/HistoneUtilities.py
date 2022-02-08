#histone related methods
import os
import sys
import time
import numpy as np
import networkx as nx
import gzip
import itertools
import math
import random
from copy import deepcopy
import HistEvalUtilities
import scipy.stats

#def paramdict2vec(paramdict,sortmarkers):
#    """parameter dictionary to vec
#    Args:
#       paramdict:
#       sortmarkers:
#    Returns:
#       objval:
#    """
#    paramvec = [paramdict[marker] for marker in sortmarkers]
#    curorder = 2
#    while len(paramvec) < len(paramdict.keys()):
#        paramvec += [paramdict[marktuple] for marktuple in itertools.combinations(sortmarkers,2)]
#        curorder += 1
#    return np.array(paramvec,dtype=np.float)   

def paramdict2vec(parammodel,params,sortmarkers,width,compcount,order):
    """param dict to vector
    Args:
       parammodel,params,sortmarkers:
       width,compcount,order:
    Returns:
    """  
    paramvec = {}
    for tpar in params.keys():
        if parammodel == "param":
           paramvec[tpar] = [params[tpar][marker][wind] for marker in sortmarkers for wind in xrange(1,width+1)]
           if order >= 2:
              paramvec[tpar].extend([params[tpar][(mark1, mark2)][wind] for mark1,mark2 in itertools.combinations(sortmarkers,2) for wind in xrange(1,width+1)]) 
        elif parammodel == "nonparam":
           paramvec[tpar] = list(itertools.chain(*[params[tpar][marker][wind] for marker in sortmarkers for wind in xrange(1,width+1)]))
    return paramvec

def getWidth(domparams):
    """returns width of paramdict
    """
    return max([curw for tpar in domparams.keys() for marker in domparams[tpar].keys() for curw in domparams[tpar][marker].keys()])


def getOrder(domparams):
    """returns the order
    """
    order = 1
    for curkey in domparams.values()[0].keys():
        if type(curkey) == tuple and len(curkey) > order:
           order = len(curkey)   
    return order

def genShuffleMarkData(chro2markinfo):
    """shuffles mark data
    Args:
       chro2markinfo:
    Returns:
    """
    modchro2markinfo = {}
    for chro in chro2markinfo.keys():
        print chro2markinfo[chro]
        print chro2markinfo[chro].keys()
        exit(1)
        for mark in chro2markinfo[chro].keys():
            pass
    return

def genPermuteTestMarkData(chro2markinfo):
    """permutes marker data
    Args:
       chro2markinfo:
    Returns:
    """
    return


def genShuffledDomain(chro2doms,nodemap):
    """generates shuffled domains
    Args:
       chro2doms,nodemap:
    Returns:
       outdoms:
    """
    outdoms = {}
    for chro,domains in chro2doms.items():
        len2dist = {}
        for start,end in domains:
            len2dist.setdefault(end-start+1,0)
            len2dist[end-start+1] += 1
        print chro
        nodecount = nodemap[chro]
        allnodes = range(1,nodecount+1)
        truepart = HistEvalUtilities.addEmptyClusters(domains,allnodes)
        seennodes = set(item for clu in truepart for item in clu)
        assert len(seennodes ^ set(allnodes)) == 0
        trueclust = sorted(truepart)
        lens = [len(clust) for clust in trueclust]
        random.shuffle(lens)
        locs = [0]
        for ind in xrange(0,len(lens)-1):
            locs.append(lens[ind]+locs[ind])
        if len(locs) > 1:    
           locs = locs[1:]
        curclust = [allnodes[0:locs[0]]]
        for ind in xrange(len(locs)-1):
            curclust.append(allnodes[locs[ind]:locs[ind+1]])
        curclust.append(allnodes[locs[-1]:nodecount])
        curclust = [clu for clu in curclust if len(clu) != 0]
        putcurclust = set((clu[0],clu[-1]) for clu in curclust)
        len2info,len2doms = {}, {}
        for start,end in putcurclust:
            len2info.setdefault(end-start+1,[])
            len2info[end-start+1].append((start,end))
        for mylen in len2info.keys():
            if not len2dist.has_key(mylen):
               continue
            blocks = len2info[mylen]
            random.shuffle(blocks)
            len2doms[mylen] = set(blocks[0:len2dist[mylen]])
        putdoms = [block for mylen in len2doms.keys() for block in len2doms[mylen]]
        outdoms[chro] = list(putdoms)
        tsum,tsum2,tsum3 = sum([end-start+1 for start,end in putcurclust]), sum([end-start+1 for start,end in putdoms]), sum([end-start+1 for start,end in domains])
        assert tsum == nodecount and tsum2 == tsum3
        seen1 = set(item for clu in curclust for item in clu)
        tseen1 = [item for clu in curclust for item in clu]
        seen2 = set(allnodes)
        seen3 = set(item for clu in trueclust for item in clu)
        assert len(seen1 ^ seen2) == 0 and len(seen1 ^ seen3) == 0 and len(seen1) == len(tseen1)
    return outdoms


def smoothAverage(marklist):
    """smoothing average
    Args:
       marklist:
    Returns:
       marklist:
    """
    tmarklist = deepcopy(marklist)
    marklist = []
    for tind,markinfo in enumerate(tmarklist):
        putdict = {}
        for marker in markinfo.keys():
            putdict[marker] = []
            countvec = [0.0] * nodecounts[tind]
            for pos,count in markinfo[marker]:
                countvec[pos-1] = count
            avgcountvec = list(countvec)
            for ind1 in xrange(len(countvec)):
                startind,endind = max(0,ind1-1), min(nodecounts[tind],ind1+2)
                avgval = np.mean([countvec[inind] for inind in xrange(startind,endind)])
                avgcountvec[ind1] = avgval
            for pos,tval in enumerate(avgcountvec):
                if tval >= 0.00000000001:
                   putdict[marker].append((pos+1,tval))
        marklist.append(deepcopy(putdict)) 
    return marklist

def modifyMarkerData(marklist,nodecounts,prepromodel,avgmode=False):
    """marker data modifier
    Args:
       marklist,nodecounts:
       prepromodel,avgmode:
    Returns:
       marklist:
    """
    assert prepromodel in ["binary","binary0.5","loglinear","linear","logit","colnorm","poisson0.99","poisson0.9"]
    if avgmode:
       marklist = smoothAverage(marklist)

    if prepromodel == "linear":
       return marklist
        
    tmarklist = deepcopy(marklist)
    marklist = []
    if prepromodel in ["poisson0.99","poisson0.9"]:
       thresval = float(prepromodel.replace("poisson",""))
       for mind,markinfo in enumerate(tmarklist):
           putdict = {}
           for marker in markinfo.keys():
               meanval = np.sum([count for pos,count in markinfo[marker]])/float(nodecounts[mind])
               putdict[marker] = [(pos,1) for pos,count in markinfo[marker] if scipy.stats.poisson.cdf(count,meanval) >= thresval]
           marklist.append(dict(putdict))
    elif prepromodel in ["binary","binary0.5"]:
       thresval = 0.0 if prepromodel == "binary" else 0.5
       for mind,markinfo in enumerate(tmarklist):
           putdict = {}
           for marker in markinfo.keys():
               counts = [count for pos,count in markinfo[marker]]
               if nodecounts[mind] != len(counts):
                  counts.extend([0.0]*(nodecounts[mind]-len(counts)))
               meanval,stdval = np.mean(counts), np.std(counts)
               putdict[marker] = [(pos,1) for pos,count in markinfo[marker] if count>=meanval+(thresval*stdval)]
           marklist.append(dict(putdict))
    elif prepromodel == "loglinear":
       for mind,markinfo in enumerate(tmarklist):
           putdict = {}
           for marker in markinfo.keys():
               putdict[marker] = [(pos,math.log(1.0+count)) for pos,count in markinfo[marker]]
           marklist.append(dict(putdict))
    elif prepromodel == "colnorm":
       sortmarkers = sorted(list(set(mark for markinfo in tmarklist for mark in markinfo.keys())))
       for wind,markinfo in enumerate(tmarklist):
           vals = np.zeros((len(sortmarkers),nodecounts[wind]),dtype=np.float64)
           for mind,marker in enumerate(sortmarkers):
               for pos,count in markinfo[marker]:
                   vals[mind,pos-1] = count
           for tnode in xrange(1,nodecounts[wind]+1):
               divsum = np.sum(vals[:,tnode-1])
               if divsum != 0.0:
                  for mind,marker in enumerate(sortmarkers):
                      vals[mind,tnode-1] /= divsum
                  assert abs(sum(vals[:,tnode-1]) -1.0) < 0.001        
           putdict = {}        
           for mind,marker in enumerate(sortmarkers):
               putdict[marker] = [(rind,vals[mind,rind-1]) for rind in xrange(1,nodecounts[wind]+1)]        
           marklist.append(dict(putdict))  
    elif prepromodel == "stddev":
          pass 
          #tmarkinfo = deepcopy(markinfo)
          #markinfo = {}
          #for marker in tmarkinfo.keys():
          #    counts = [count for pos,count in tmarkinfo[marker]]
          #    meanval,stdval = np.mean(counts), np.std(counts)
          #    markinfo[marker] = [(pos,(count-meanval)/stdval) for pos,count in tmarkinfo[marker]]
    return marklist


def logSumExp(xs,NEGINF=-1.0e50):
    """log sum of exponentials
    Args:
       xs: list of expo coefs
    Returns:
       sumval:  
    """
    if len(xs) == 1: 
       return xs[0]
    maxval = max(xs);
    if maxval == NEGINF:
       return NEGINF
    sumval = sum(math.exp(item-maxval) for item in xs if item > NEGINF)
    #return max(maxval+(math.log(sumval) if sumval != 0.0 else NEGINF), NEGINF)
    return maxval+(math.log(sumval) if sumval != 0.0 else NEGINF)


def getEmptyClusters(domains,allnodes):
    """gets empty clusters
    Args:
       domains,allnodes:
    Returns:
       emptydoms: empty domains
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
    part2 = [part for part in part2 if len(part)>0]
    part2 = [(part[0],part[-1]) for part in part2]  
    return part2 


def getCumuls(node2start,node2end,node2notstart={},node2notend={}):
    """get cumulative dist
    Args:
       node2start,node2end,node2notstart,node2notend:
    Returns:
    """
    cumnode2start = getCumulative(node2start)
    cumnode2end = getCumulative(node2end)
    cumnode2notstart = getCumulative(node2notstart)
    cumnode2notend = getCumulative(node2notend)
    return cumnode2start, cumnode2end, cumnode2notstart, cumnode2notend


def makeInterMatrix(locs,resol):
    """makes interaction matrix from raw data
    Args:
       locs:
       resol:
    Returns:
       freqmat:
       in2loc: 
    """
    maxind = max(item/resol for items in locs for item in items[0:2])
    freqmat = np.zeros((maxind+1,maxind+1), dtype=np.float)
    in2loc = {ind:(ind,ind+1) for ind in xrange(maxind+1)}
    for pos1,pos2,count in locs:
        ind1,ind2 = int(pos1/resol), int(pos2/resol)
        freqmat[ind1,ind2] += count 
        if ind1!=ind2:
           freqmat[ind2,ind1] += count
    return freqmat,in2loc
       

def readBedFile(fpath):
    """reads bed file returns set of regions
    Args:
       fpath:
    Returns:
       chro2vals:
    """
    chro2vals = {}
    if fpath.endswith(".gz"): 
       opmethod = gzip.open(fpath,"r")
    else:
       opmethod = open(fpath,"r") 
    with opmethod as infile:
        for line in infile:
            chrostr,startstr,endstr = line.rstrip().split("\t")[0:3]
            chro = chrostr.replace("chr","")
            start,end = int(startstr), int(endstr)
            chro2vals.setdefault(chro,set())
            chro2vals[chro].add((start,end)) 
    return chro2vals


def readWigFile(fpath,sentchro=None):
    """reads wig file, checks extension
    Args:
       fpath:
       sentchro:
    Returns:
       vals:   
    """
    chro, start, step, outmode = None, None, None, None
    vals,curcount = {},0
    if fpath.endswith(".gz"): 
       opmethod = gzip.open(fpath,"r")
    else:
       opmethod = open(fpath,"r") 
    with opmethod as infile:
        for line in infile:
            line = line.rstrip()
            if not line.startswith("track"):
               if line.startswith("fixedStep"):
                  assert outmode != "variable" 
                  outmode = "fixed"  
                  chrostr,startstr,stepstr = line.split()[1:-1]
                  chro = chrostr.replace("chrom=chr","")
                  if sentchro!=None and sentchro != chro:
                     continue 
                  start,step = int(startstr.replace("start=","")), int(stepstr.replace("step=",""))
                  curcount = 0
                  vals.setdefault(chro,set())
               elif line.startswith("variableStep"):
                  assert outmode != "fixed"  
                  outmode = "variable"  
                  chrostr = line.split()[1]
                  chro = chrostr.replace("chrom=chr","")
                  if sentchro!=None and sentchro != chro:
                     continue 
                  vals.setdefault(chro,set())   
               elif outmode == "fixed":
                  if sentchro!=None and sentchro != chro:
                     continue  
                  if int(line) != 0:
                     vals[chro].add((start+(curcount*step)+(step/2),int(line)))
                  curcount += 1
               elif outmode == "variable":
                  if sentchro!=None and sentchro != chro:
                     continue   
                  start,count = [int(item) for item in line.split("\t")] 
                  if count != 0:
                     vals[chro].add((start,count))
    return vals


def sol2dict(sentcoefs,sortmarkers,width,infermodel,classcount=None,compcount=None):     
    """vec to dict params
    Args:
       sentcoefsm,sortmarkers,width:
       infermodel,classcount: classcount only for crflatent model
       compcount:
    Returns:
       paramdict:
    """
    paramdict = {}
    if infermodel in ["single-memm","single-memm2"]:
       vallist = [("term",sentcoefs[0])] 
    elif infermodel in ["crf","semicrf","pseudo"]:
       vallist = [("bound",sentcoefs[0]),("inside",sentcoefs[1]),("empty",sentcoefs[2])]
    elif infermodel == "crflatent":
       vallist = [("bound{0}".format(cind+1),sentcoefs[2*cind]) for cind in xrange(classcount)] + [("inside{0}".format(cind+1),sentcoefs[2*cind+1]) for cind in xrange(classcount)] + [("empty",sentcoefs[2*classcount])]       
    ucompcount = 1 if compcount == None else compcount  
    for curpar, solx in vallist:
        paramdict[curpar] = {marker: {wind+1: solx[ind*width*ucompcount+(wind*ucompcount):ind*width*ucompcount+((wind+1)*ucompcount)] for wind in xrange(width)} for ind,marker in enumerate(sortmarkers)}
        curorder,startcount = 2, len(sortmarkers)*width*ucompcount
        while len(paramdict[curpar].keys()) < len(solx)/(width*ucompcount):
            paramdict[curpar].update({marktuple: {wind+1: solx[startcount+ (mind*width*ucompcount)+(wind*ucompcount):startcount+(mind*width*ucompcount)+((wind+1)*ucompcount)] for wind in xrange(width)} for mind,marktuple in enumerate(itertools.combinations(sortmarkers,curorder))})
            curorder += 1
            startcount += len(list(itertools.combinations(sortmarkers,curorder)))                                                                         
    return paramdict


def processMarkData(markinfo):
    """processes marker data
    Args:
       markinfo:
    Returns:
       mark2pos2count:
    """
    mark2pos2count = {marker:{} for marker in markinfo.keys()}
    for marker in markinfo.keys():
        for pos,count in markinfo[marker]:
            mark2pos2count[marker][pos] = count
    return mark2pos2count


def getLocs(sorteddoms,nodecount):
    """get locations
    Args:
       sorteddoms:
       nodecount:
    Returns:
       boundlocs,indomlocs,remlocs:
    """
    boundlocs = set(tind for tdom in sorteddoms for tind in tdom)
    indomlocs = set(tind for start,end in sorteddoms for tind in xrange(start+1,end))
    remlocs = set(range(1,nodecount+1)).difference(indomlocs | boundlocs)
    return boundlocs,indomlocs,remlocs





def getCumulative(node2negprob):
    """cumulative distribution
    Args:
       node2negprob:
    Returns:
       cumnode2negprob: 
    """
    if len(node2negprob.keys()) == 0:
       return None 
    maxnode = max(node2negprob.keys())
    cumnode2negprob = {0: 0.0, 1: node2negprob[1]}
    for node in xrange(2,maxnode+1):
        cumnode2negprob[node] = cumnode2negprob[node-1] + node2negprob[node] 
    return cumnode2negprob

def getCompPars(retvec,compcount):
    """gets compcount params
    Args:
       retvec,compcount:
    Returns:
       countvec:
    """
    countvec = []
    for item in retvec:
        if item < 0.01:
           countvec.extend([0.0]*compcount)
           continue        
        countvec.append(item)
        for ind in xrange(compcount-1):
            #countvec.append(math.pow(item,1.0+(1.0*float(ind+1))/(compcount-1))) 
            countvec.append(math.pow(item,1.0+(1.0*float(ind+1))/(compcount-1)))     
            #countvec.append(countvec[-1]*item)        
    return np.array(countvec)


def getCountVec(mark2pos2count,node,sortmarkers,width,order):
    """get count vector
    Args:
       mark2pos2count,node:
       sortmarkers,width,order:
    Returns:
       countvec:
    """
    precountvec = [mark2pos2count[marker][node+win] if mark2pos2count[marker].has_key(node+win) else 0.0 for marker in sortmarkers for win in xrange(-1*width+1,width)]
    countvec = []
    for mind,marker in enumerate(sortmarkers):
        countvec.append(precountvec[(2*width-1)*mind + width-1])
        for win in xrange(1,width):
            countvec.append(precountvec[(2*width-1)*mind + width-1 + win] + precountvec[(2*width-1)*mind + width-1 - win])
    if order > 1:
       countvec += [countvec[(width*ind1)+win]*countvec[(width*ind2)+win] for ind1,ind2 in itertools.combinations(range(len(sortmarkers)),2) for win in xrange(width)]
    return countvec


def writeDomainFile(domoutfile,domains):
    """writes domain file
    Args:
       domoutfile:
       domains:
       comp2scale:
    Returns:
    """
    with open(domoutfile,"w") as outfile:
        for start,end in domains:
            outfile.write("{0},{1}\n".format(start,end))


def readMultiDomainFile(domfile):
    """reads domain file of list
    Args:
       domfile:
    Returns:
       domainlist,nodecounts:
    """
    domainlist,curdomain,nodecounts = [], [],[]
    with open(domfile,"r") as infile:
        for line in infile:
            line = line.rstrip()
            if line == "":
               domainlist.append(list(curdomain))
               curdomain = []
               continue
            if line.startswith("nodecount"):
               nodecounts.append(int(line.split("\t")[1])) 
               continue 
            curdomain.append(tuple([int(item) for item in line.split(",")]))
    domainlist.append(list(curdomain))
    return domainlist,nodecounts

def writeMultiDomainFile(domoutfile,domainlist,nodecounts):
    """writes domain file of list
    Args:
       domoutfile:
       domainlist: list of domain partition
       nodecounts:
    Returns:
    """
    with open(domoutfile,"w") as outfile:
         outfile.write("\n\n".join("nodecount\t{0}\n".format(nodecounts[ind]) + "\n".join("{0},{1}".format(start,end) for start,end in domain) for ind,domain in enumerate(domainlist)))


def writeMultiscaleDomainFile(scale2doms,outfpath):
    """write multiscale domain file
    Args:
       scale2dms:
       outfpath:
    Returns:
    """
    with open(outfpath,"w") as outfile:
        for scale in scale2doms.keys():
            outfile.write("scale\t{0}\n".format(scale))
            for start,end in scale2doms[scale]:
                outfile.write("{0},{1}\n".format(start,end))
         
def readMultiscaleDomainFile(domfile):
    """reads multiscale domain file
    Args:
       domfile:
    Returns:
       scale2doms:
    """
    scale2doms, curscale = {}, None
    with open(domfile,"r") as infile:
        for line in infile:
            line = line.rstrip()
            if line.startswith("scale"):
               curscale = float(line.split("\t")[1])
               scale2doms[curscale] = []
            else:
               splitted = line.split(",")
               scale2doms[curscale].append((int(splitted[0]),int(splitted[1])))
    return scale2doms

            
def readDomainFile(domfile):
    """reads domain file
    Args:
       domfile:
    Returns:
       domainlist:
    """
    domainlist, doms = [], []
    with open(domfile,"r") as infile:
        for line in infile:
            line = line.rstrip()
            if line == "":
               domainlist.append(list(doms))
               doms = []
               continue
            splitted = line.split(",")
            doms.append((int(splitted[0]),int(splitted[1])))
    domainlist.append(list(doms))        
    return domainlist


def readPosFile(fpath):
    """reads pos file
    Args:
       fpath:
    Returns:
       locs:   
    """
    with open(fpath,"r") as infile:
        locs = [(float(item) for item in line.rstrip().split("\t")) for line in infile]
    return locs

def writePosFile(locs,fpath):
    """writes position file
    Args:
       locs:
       fpath:
    Returns:
    """
    with open(fpath,"w") as outfile:
        outfile.write("\n".join(["\t".join(loc) for loc in locs]))


def readMarkerFile(fpath):
    """reads marker count file
    Args:
       fpath:
    Returns:
       marklist: 
    """
    marklist,markinfo = [], {}
    with open(fpath,"r") as infile:
        for line in infile:
            line = line.rstrip()
            if line == "":
               marklist.append(dict(markinfo))
               markinfo = {}
               continue
            marker, pos, count = line.split("\t")
            pos,count = int(pos), float(count)
            markinfo.setdefault(marker,set())
            markinfo[marker].add((pos,count))
    marklist.append(dict(markinfo))
    return marklist


def writeMarkerFile(marklist,fpath):
    """writes marker file
    Args:
       marklist:
       fpath:
    Returns:
    """
    with open(fpath,"w") as outfile:
        outfile.write("\n".join("\n".join("{0}\t{1}\t{2}".format(marker,pos,count) for marker in markinfo.keys() for pos,count in markinfo[marker]) + "\n" for markinfo in marklist))

               
def writeMetaFile(metadata,objoutfile):
    """writes meta file
    Args:
       metadata:
       objoutfile:
    Returns:
    """
    with open(objoutfile,"w") as outfile:
        for key in metadata.keys():
            if type(metadata[key]) == list:
               keystr = "\t".join([str(item) for item in metadata[key]])
            else:
               keystr = metadata[key]    
            outfile.write("{0}\t{1}\n".format(key,keystr))
            
def readMetaFile(deconfile):
    """read meta file
    Args:
       deconfile:
    Returns:
       metata:
    """
    metadata = {}
    with open(deconfile,"r") as infile:
        for line in infile:
            splitted = line.rstrip().split("\t")
            if len(splitted) == 2:
               metadata[splitted[0]] = float(splitted[1])
            else:
               metadata[splitted[0]] = [float(item) for item in splitted[1:]]
    return metadata


def readDomainParamFile(parampath):
    """read domain param file
    Args:
       parampath:
    Returns:
       paramdict:
    """
    def processItem(item):
        """
        """
        if item.find(",") != -1:
           return [float(tcur) for tcur in item.split(",")]
        return float(item)
    paramdict, curkey = {}, None
    with open(parampath,"r") as infile:
        for line in infile:
            line = line.rstrip()
            foundval = any([True if line == item else False for item in ["start","end","term","empty","inter"]]+[True if line.startswith(item) else False for item in ["bound","inside"]])
            if foundval:
               curkey = line
               paramdict[curkey] = {}
               continue
            splitted = line.split("\t")
            if len(splitted) == 3:
               paramdict[curkey].setdefault(splitted[0], {})
               paramdict[curkey][splitted[0]][int(splitted[1])] = processItem(splitted[-1])
            elif len(splitted) >= 4:
               paramdict[curkey].setdefault(tuple(splitted[0:-2]), {}) 
               paramdict[curkey][tuple(splitted[0:-2])][int(splitted[-2])] = processItem(splitted[-1])
    return paramdict

def readNodecount(fpath):
    """
    """
    with open(fpath,"r") as infile:
       for line in infile:
           return int(line.rstrip())

def writeNodecount(fpath,nodecount):
    """
    """
    with open(fpath,"w") as outfile:
       outfile.write("{0}\n".format(nodecount)) 

    
def writeDomainParamFile(parampath,paramdict):
    """write domain param file
    Args:
       parampath:
       paramdict:
    Returns:
    """
    #model = "linear"
    #if type(paramdict.values()[0].values()[0].values()[0]) in [np.ndarray, np.array, list]:
    #   model = "nonparam"    
    singles = set(param for probparam in paramdict.keys() for param in paramdict[probparam].keys() if type(param) != tuple)
    multis = set(param for probparam in paramdict.keys() for param in paramdict[probparam].keys() if type(param) == tuple)
    with open(parampath,"w") as outfile:
        for pparam in paramdict.keys(): 
            outfile.write("{0}\n".format(pparam)) 
            #if model == "linear":
            #   outfile.write("\n".join(["{0}\t{1}\t{2}".format(single,width,paramdict[pparam][single][width]) for single in singles for width in paramdict[pparam][single].keys()]) + "\n")
            #   outfile.write("".join(["\t".join(list(partuple) + [str(width), str(paramdict[pparam][partuple][width])]) + "\n" for partuple in multis for width in paramdict[pparam][partuple].keys()]))              
            #elif model == "nonparam":
            outfile.write("\n".join(["{0}\t{1}\t{2}".format(single,width,",".join(str(citem) for citem in paramdict[pparam][single][width])) for single in singles for width in paramdict[pparam][single].keys()]) + "\n")
            outfile.write("".join(["\t".join(list(partuple) + [str(width), ",".join(str(citem) for citem in paramdict[pparam][partuple][width])]) + "\n" for partuple in multis for width in paramdict[pparam][partuple].keys()]))              


def readHawkesParamFile(parampath):
    """read hawkes param file
    Args:
       parampath:
    Returns:
       mudict:
       alphadict:
       kerntype,kernmeans:
    """
    mudict,alphadict,kerntype,kernmeans,mumode,curmode = {}, {}, None, None, None
    with open(parampath,"r") as infile:
        for ind,line in infile:
            line = line.rstrip()
            if line.startswith("alpha"):
               curmode = "alpha"
               continue
            if line.startswith("mu"):
               curmode = "mu"
               continue
            if ind == 0:
               kerntype = line.split("\t")[1]
            elif ind == 1:
               kernmeans = [float(item) if item != 'None' else None for item in line.split("\t")[1:]]
            elif curmode == "mu":
               splitted = line.split("\t")
               mudict[splitted[0]] = float(splitted[1]) 
            elif curmode == "alpha":
               splitted = line.split("\t") 
               alphadict[tuple(splitted[0:2])] = [float(item) for item in splitted[2:]] 
    return mudict,alphadict,kerntype,kernmeans

    
def writeHawkesParamFile(parampath,mudict,alphadict,kerntype="gauss",kernmeans=None):
    """write hawkes param file
    Args:
       parampath:
       mudict:
       alphadict:
       kerntype: kernel type
       kernmeans: if None 
    Returns:
    """
    with open(parampath,"w") as outfile:
        outfile.write("kerntype\t{0}\n".format(kerntype))
        outfile.write("kernmeans\t{0}\n".format("\t".join(kernmeans)))
        outfile.write("mu\n" + "\n".join(["{0}\t{1}".format(marker,mudict[marker]) for marker in mudict.keys()]) + "\n")
        outfile.write("alpha\n" + "\n".join(["{0}\t{1}\t{2}".format(mark1,mark2,"\t".join(vallist)) for (mark1,mark2),vallist in alphadict.items()]) + "\n")


        
