#Domain Finder as DP
import numpy as np
import os
import sys
sys.path.append("lib")
import HistoneUtilities
import myutilities as myutil
import gzip
import itertools
import time
import math
import random
#import estCRFParamsLatent
#import estCRFParams
from copy import deepcopy
import operator
import cPickle as cpickle
sys.path.append("Tests")
from TestDomainFinder import TestDomainFinder


def estWeightsApproxTriple(nodecount,markinfo,params,sortmarkers,domprior,width,parammodel,infermodel,order,compcount):
    """
    """
    def runParam(cnode):
        return np.array(HistoneUtilities.getCountVec(mark2pos2count,cnode,sortmarkers,width,order),dtype=np.float)
    runfunc = "run{0}".format(parammodel.capitalize()) 
    paramvec = HistoneUtilities.paramdict2vec(parammodel,params,sortmarkers,width,compcount,order)
    mark2pos2count = HistoneUtilities.processMarkData(markinfo)
    
    node2bound,node2empty,node2inside, countvecs = {}, {}, {}, {}
    for node in xrange(1,nodecount+1):
        countvecs[node] = np.array(locals()[runfunc](node))
    for node in xrange(1,nodecount+1):
        node2bound[node] = np.dot(countvecs[node],paramvec["bound"])
        node2inside[node] = np.dot(countvecs[node],paramvec["inside"])
        node2empty[node] = np.dot(countvecs[node],paramvec["empty"])
    prob2bound,prob2empty = {}, {}
    for node in xrange(1,nodecount+1):
        prob2bound[node] = node2bound[node] - HistoneUtilities.logSumExp([node2bound[node],node2inside[node],node2empty[node]])
        prob2empty[node] = node2empty[node] - HistoneUtilities.logSumExp([node2bound[node],node2inside[node],node2empty[node]])
    L = 40
    newvec, block2inside = {}, {}
    for node in xrange(1,nodecount+1):
        newvec[(node,node)] = np.array(countvecs[node])
        block2inside[(node,node)] = np.dot(newvec[(node,node)],paramvec["inside"])  
    for mylen in xrange(2,L+2):
        for start in xrange(1,nodecount-mylen+2):
            end = start+mylen-1
            newvec[(start,end)] = np.array(newvec[(start,end-1)]) + newvec[(end,end)]
            block2inside[(start,end)] = np.dot(newvec[(start,end)],paramvec["inside"]) / float(end-start+1)
            #block2inside[(start,end)] = np.dot(newvec[(start,end)],paramvec["inside"])
            #newvec[(start,end)] = np.array(newvec[(start,end-1)])
            #for lind,item in enumerate(newvec[(end,end)]):
            #    if newvec[(start,end)][lind] == 0.0 and item == 1.0:
            #       newvec[(start,end)][lind] = 1.0
            #node2inside[(start,end)] = np.dot(newvec[(start,end)],paramvec["bound"])      
            #node2inside[(start,end)] = np.dot(newvec[(start,end)],paramvec["bound"]) / float(end-start+1) #comment maybe     
    prob2inside = {}
    for mylen in xrange(1,L+1):      
        for start in xrange(1,nodecount+2-mylen):
            end = start+mylen-1
            sumval = 0.0
            for tpos in xrange(start,end+1):
                sumval -= HistoneUtilities.logSumExp([block2inside[(start,end)],node2bound[tpos],node2empty[tpos]])
            prob2inside[(start,end)] = ((end-start+1) * block2inside[(start,end)]) + sumval 
                      
    domprobs,domnotprobs = np.full((nodecount+1, nodecount+1),-10000000000000.0), np.zeros((nodecount+1,nodecount+1),dtype=np.float)
    for node in xrange(1,nodecount+1):
        domnotprobs[node,node] = prob2empty[node]
    for node in xrange(1,nodecount):
        domprobs[node,node+1] = prob2bound[node] + prob2bound[node+1]       
    for domlen in xrange(2,nodecount+1):
        for start in xrange(1,nodecount-domlen+2):
            end = start+domlen-1
            domnotprobs[start,end] = domnotprobs[start,end-1] + prob2empty[end]
        if domlen > 2:
           for start in xrange(1,nodecount-domlen+2):
               end = start+domlen-1
               if prob2inside.has_key((start+1,end-1)):
                  domprobs[start,end] = prob2bound[start] + prob2bound[end] + prob2inside[(start+1,end-1)]
    weights = {"pos":domprobs, "neg":domnotprobs}
    if TESTMODE:
       for node in prob2bound.keys():
           assert prob2bound[node] < 0.0001
       for node in prob2empty.keys():
           assert prob2empty[node] < 0.0001
       for info in prob2inside.keys():
           assert prob2inside[info] < 0.0001
    return weights 



def estCRFwe1NOTDONE(node2bound,node2inside,node2empty,nodecount):
    """
    """   
    node2val = {}
    for node in xrange(1,nodecount+1):
        countvec = locals()[runfunc](node)
        node2val[node] = np.array(countvec) 
    countdict = {(tnode,tnode): np.array(node2val[tnode])  for tnode in xrange(1,nodecount+1)}
    for tnode in xrange(2,nodecount+1):
        countdict[(tnode,tnode-1)] = np.array([0.0]*len(node2val[tnode])) 
    for domlen in xrange(2,nodecount+1):
        for start in xrange(1,nodecount-domlen+2):
            countdict[(start,start+domlen-1)] = countdict[(start,start+domlen-2)] + node2val[start+domlen-1] 
    for tnode1 in xrange(1,nodecount+1):
        for tnode2 in xrange(tnode1,nodecount+1):
            for k in xrange(len(countdict[(tnode1,tnode2)])):
                countdict[(tnode1,tnode2)][k] = 1.0 if countdict[(tnode1,tnode2)][k] >= 1.0 else 0.0
    domprobs,domnotprobs,dominprobs = [np.zeros((nodecount+1,nodecount+1),dtype=np.float) for ind in xrange(3)]  
    for tnode1 in xrange(1,nodecount+1):
        for tnode2 in xrange(tnode1,nodecount+1):  
            if tnode1 != tnode2:
               domprobs[tnode1,tnode2] = np.dot(paramvec["bound"],countdict[(tnode1,tnode2)]) 
            dominprobs[tnode1,tnode2] = np.dot(paramvec["inside"], countdict[(tnode1,tnode2)]) 
            domnotprobs[tnode1,tnode2] = np.dot(paramvec["empty"], countdict[(tnode1,tnode2)])           
    for start in xrange(1,nodecount+1):
        for end in xrange(start+1,nodecount+1):
            domprobs[start,end] += dominprobs[start+1,end-1]
    return domprobs,domnotprobs

def estCRFweNorm(node2bound,node2inside,node2empty,nodecount):
    """ normalized weight estimate
    """
    domprobs,domnotprobs,dominprobs = [np.zeros((nodecount+1,nodecount+1),dtype=np.float) for ind in xrange(3)]   
    for tnode in xrange(1,nodecount):
        domnotprobs[tnode,tnode] = node2empty[tnode]
    for start in xrange(1,nodecount):
        domprobs[start,start+1] = node2bound[start] + node2bound[start+1] 
        domnotprobs[start,start+1] = node2empty[start] + node2empty[start+1] 
    for domlen in xrange(3,nodecount+1):
        for start in xrange(1,nodecount-domlen+2):
            domprobs[start,start+domlen-1] = domprobs[start,start+domlen-2] + node2bound[start+domlen-1] - node2bound[start+domlen-2] 
            dominprobs[start+1,start+domlen-2] = dominprobs[start+1,start+domlen-3] + node2inside[start+domlen-2] 
            domnotprobs[start,start+domlen-1] = domnotprobs[start,start+domlen-2] + node2empty[start+domlen-1] 
    domprobs /= 2.0
    for start in xrange(1,nodecount+1):
       for end in xrange(start+1,nodecount+1):
           dominprobs[start,end] /= (end-start+1)
    for start in xrange(1,nodecount+1):
       for end in xrange(start+1,nodecount+1):
           domnotprobs[start,end] /= (end-start+1) 
    for start in xrange(1,nodecount+1):
       for end in xrange(1,start):
           assert dominprobs[(start,end)] == 0.0         
    for start in xrange(1,nodecount+1):
       for end in xrange(start+1,nodecount+1):
           domprobs[(start,end)] += dominprobs[(start+1,end-1)]
    return domprobs,domnotprobs 

def estclassicCRF(node2bound,node2inside,node2empty,nodecount):
    """estimates crf weights for classical case
    Args:
       node2bound,node2inside,node2empty: coefficients to be used
       nodecount: total number of nodes
    """
    L = 100
    domprobs,domnotprobs = [np.zeros((nodecount+1,nodecount+1),dtype=np.float) for ind in xrange(2)]   
    for tnode in xrange(1,nodecount):
        domnotprobs[tnode,tnode] = node2empty[tnode]
    for start in xrange(1,nodecount):
        domprobs[start,start+1] = node2bound[start] + node2bound[start+1] 
        domnotprobs[start,start+1] = node2empty[start] + node2empty[start+1] 
    for domlen in xrange(3,nodecount+1):
        for start in xrange(1,nodecount-domlen+2):
            if domlen <= L:
               domprobs[start,start+domlen-1] = domprobs[start,start+domlen-2] + node2bound[start+domlen-1] - node2bound[start+domlen-2] + node2inside[start+domlen-2]  
            else:
               domprobs[start,start+domlen-1] = -1000000000000.0
            domnotprobs[start,start+domlen-1] = domnotprobs[start,start+domlen-2] + node2empty[start+domlen-1]
    return domprobs,domnotprobs

def getCompParsBernstein(datavec,basecount):
    """get component parts by bernstein
    Args:
       datavec,basecount:
    Returns:
       sentvec:
    """         
    sentvec = []
    for mval in datavec:
        usevec = [(1-mval)**(basecount-1)]
        for bind in xrange(1,basecount):
            usevec.append(usevec[bind-1]*(basecount-bind)*mval/(bind*(1.0-mval)))
        sentvec.extend(usevec)
    return np.array(sentvec)
        
def estWeightsCrf(nodecount,markinfo,params,sortmarkers,domprior,width,parammodel,infermodel,order,compcount):
    """estimates the weights for crf
    Args:
       nodecount,markinfo:
       params: domain parameter dictionary
       sortmarkers,domprior: length prior
       width,parammodel,infermodel,order,compcount,basecount:
    Returns:
       weights,fixedval:[node2bound,node2inside,cumnode2bound,cumnode2inside]:
    """
    def runNonparam(cnode):
        retvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,cnode,sortmarkers,width,order),dtype=np.float)
        return getCompParsBernstein(retvec,compcount)
    def runParam(cnode):
        return np.array(HistoneUtilities.getCountVec(mark2pos2count,cnode,sortmarkers,width,order),dtype=np.float)
    runfunc = "run{0}".format(parammodel.capitalize())
    paramvec = HistoneUtilities.paramdict2vec(parammodel,params,sortmarkers,width,compcount,order)
    mark2pos2count = HistoneUtilities.processMarkData(markinfo)
    node2bound,node2inside,node2empty = {}, {}, {}
    for node in xrange(1,nodecount+1):
        countvec = locals()[runfunc](node)
        node2bound[node] = np.dot(paramvec["bound"], countvec)
        node2inside[node] = np.dot(paramvec["inside"], countvec) 
        node2empty[node] = np.dot(paramvec["empty"], countvec)
    domprobs,domnotprobs = estclassicCRF(node2bound,node2inside,node2empty,nodecount)
    weights = {"pos":np.array(domprobs), "neg":np.array(domnotprobs)} 
    return weights


def dict2paramvec(paramvec,classcount,infermodel):
    """dictionary to parameter vector
    Args:
       paramvec,classcount,infermodel:
    Returns:
       optparamx: 
    """
    optparamx = []
    if infermodel == "crflatent":
       for curclass in xrange(classcount):
           for keystr in ["bound","inside"]:
               usestr = "{0}{1}".format(keystr,curclass+1)
               optparamx.extend(paramvec[usestr])
       optparamx.extend(paramvec["empty"])        
    elif infermodel == "crf":
       for keystr in ["bound","inside","empty"]:
           optparamx.extend(paramvec[keystr])
    return optparamx


def estWeightsCrfLatent(nodecount,markinfo,params,sortmarkers,domprior,width,parammodel,order):
    """est weights crf latent case
    Args:
       nodecount,markinfo,domparams,sortmarkers,domprior,width,parammodel,order:
    Returns:
    """
    classcount = max([int(keystr.replace("bound","")) for keystr in params.keys() if keystr.find("bound")!=-1])
    paramvec = HistoneUtilities.paramdict2vec(parammodel,params,sortmarkers,width,1,order)
    optparamx = dict2paramvec(paramvec,classcount,"crflatent")
    domlen = len(domparams["empty"])
    node2bound, node2inside, node2empty = {},{},{}
    mark2pos2count = HistoneUtilities.processMarkData(markinfo)
    for cnode in xrange(1,nodecount+1):
        countvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,cnode,sortmarkers,width,order),dtype=np.float)
        for curclass in xrange(classcount):
            node2bound[(cnode,curclass)] = np.dot(optparamx[2*curclass*domlen:(2*curclass+1)*domlen], countvec) 
            node2inside[(cnode,curclass)] = np.dot(optparamx[(2*curclass+1)*domlen:(2*curclass+2)*domlen], countvec) 
        node2empty[cnode] = np.dot(optparamx[2*classcount*domlen:(2*classcount+1)*domlen], countvec)
    weights = {"pos":np.zeros((nodecount+1,nodecount+1),dtype=np.float), "neg":np.zeros((nodecount+1,nodecount+1),dtype=np.float)}
    for tnode in xrange(1,nodecount+1):
        weights["neg"][tnode,tnode] = node2empty[tnode]
    curposarr = {curclass: np.zeros((nodecount+1,nodecount+1),dtype=np.float) for curclass in xrange(classcount)}    
    for start in xrange(1,nodecount):
        for curclass in xrange(classcount):
            curposarr[curclass][start,start+1] = node2bound[(start,curclass)] + node2bound[(start+1,curclass)] 
        weights["neg"][start,start+1] = node2empty[start] + node2empty[start+1]
    for mylen in xrange(3,nodecount+1):
        for start in xrange(1,nodecount-mylen+2):
            end = start+mylen-1
            if mylen > 100:
               for curclass in xrange(classcount):
                   curposarr[curclass][start,end] = -1000000000000.0
            else:    
               for curclass in xrange(classcount):
                   curposarr[curclass][start,end]=curposarr[curclass][start,end-1]+node2bound[(end,curclass)]-node2bound[(end-1,curclass)]+node2inside[(end-1,curclass)]
            weights["neg"][start,end] = weights["neg"][start,end-1] + node2empty[end]
    for start in xrange(1,nodecount+1):
        for end in xrange(start+1,nodecount+1):
            weights["pos"][start,end] = max([curposarr[curclass][start,end] for curclass in xrange(classcount)])
    return weights
        
    
def estWeightsMemm(nodecount,markinfo,params,sortmarkers,domprior,width,parammodel,infermodel,order,compcount):
    """estimates the weights for memm model
    Args:
       nodecount,markinfo:
       params: domain parameter dictionary
       sortmarkers,domprior: length prior
       width,parammodel,infermodel,order,compcount:
    Returns:
       node2term,node2notterm:
    """
    estfunc = "est{0}".format(infermodel.replace("-","").capitalize())
    runfunc = "run{0}".format(parammodel.capitalize())
    paramvec = HistoneUtilities.paramdict2vec(parammodel,params,sortmarkers,width,compcount,order)
    mark2pos2count = HistoneUtilities.processMarkData(markinfo)
    node2term,node2notterm = {}, {}
    for node in xrange(1,nodecount+1):
        def runNonparam():
            retvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,width,order),dtype=np.float) 
            return HistoneUtilities.getCompPars(retvec,compcount)
        def runParam():
            return np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,width,order),dtype=np.float)  
        countvec = locals()[runfunc]()
        def estSinglememm(): 
            dotval = np.dot(paramvec["term"], countvec)
            node2term[node] = dotval - math.log(1.0+math.exp(dotval)) if dotval <= 20 else 0.0 # math.log(nweight/(1.0+nweight))
            node2notterm[node] = -1.0 * math.log(1.0+math.exp(dotval)) if dotval <= 20 else -1.0 * dotval
        def estSinglememm2(): 
            estSinglememm()    
        locals()[estfunc]()   
    return node2term,node2notterm 
    #fixedval = 2*sum(node2notstart.values()) if infermodel == "double" else sum(node2notstart.values())
    #if TESTMODE:
    #   paramlist = [node2term,node2notterm,cumnode2term,cumnode2notterm]   
    #   return weights,fixedval,paramlist
    #else:
    #   return weights,fixedval, None
    

def addPriorCoef(domprior,weights):
    """adds prior coefs
    Args:
       domprior,weights:
    Returns:
       weights:
    """             
    if domprior not in [None, 'None']:
       priorfunc, priorcoef = domprior
       assert priorfunc in ["geometric","powerlaw"]
       if priorfunc == "geometric":
          prifunc = lambda domlen,coef: (domlen-1)*math.log(1.0-coef) + math.log(coef)
       elif priorfunc == "powerlaw": #zips's kind
          prifunc = lambda domlen,coef: math.log(max(0.0000000000000001,math.pow(domlen,-1.0*coef) - math.pow(domlen+1,-1.0*coef))) #p(x) = ax^(-a-1) f(x) = -x^(-a)
       for node1,node2 in weights.keys():
           assert math.pow(node2-node1+1,-1.0*priorcoef) - math.pow(node2-node1+2,-1.0*priorcoef) >= 0
           weights[(node1,node2)] += prifunc(node2-node1+1,priorcoef)      
    return weights


def CRFInferLIMITED(weights,nodecount):
    """crf infer with limited #partitions
    Args:
       weights,nodecount:
    Returns:   
    """
    PARTCOUNT = 300
    objvals = {"e": np.full((nodecount+1,PARTCOUNT+1),-1000000000000000.0), "d": np.full((nodecount+1,PARTCOUNT+1),-1000000000000000.0)}
    optsols = {"e": [[[] for pind in xrange(PARTCOUNT+1)] for node in xrange(nodecount+1)], "d": [[[] for pind in xrange(PARTCOUNT+1)] for node in xrange(nodecount+1)]}
    for pind in xrange(PARTCOUNT):
        objvals["e"][0,0] = 0.0
        objvals["d"][0,0] = 0.0
    objvals["e"][1,0] = weights["neg"][1,1]
    for node in xrange(2,nodecount+1):
        if random.random() < 0.01:
           print node 
        for pind in xrange(1,min(PARTCOUNT+1,node/2+1)):
            maxweight, start, cursol,tsign = -1.0e20, None, None, None
            for prenode in xrange(max(1,node-61),node):
                tvalsum = objvals["d"][prenode-1,pind-1] + weights["pos"][prenode,node]
                if tvalsum > maxweight:
                   maxweight = tvalsum
                   cursol = [(prenode,node)]
                   start = prenode-1
                   tsign = "d"
                tvalsum = objvals["e"][prenode-1,pind-1] + weights["pos"][prenode,node]       
                if tvalsum > maxweight:
                   maxweight = tvalsum
                   cursol = [(prenode,node)]
                   start = prenode-1
                   tsign = "e"
            #assert start != None and cursol != None
            if start!=None and cursol!=None:
               objvals["d"][node,pind] = maxweight
               optsols["d"][node][pind] = optsols[tsign][start][pind-1] + cursol
    
            maxweight, start, cursol,tsign = -1.0e20, None, None, None
            for prenode in xrange(max(1,node-61),node+1):
                tvalsum = objvals["d"][prenode-1,pind] + weights["neg"][prenode,node]
                if tvalsum > maxweight:
                   maxweight = tvalsum
                   cursol = []
                   start = prenode-1
                   tsign = "d"
                tvalsum = objvals["e"][prenode-1,pind] + weights["neg"][prenode,node]       
                if tvalsum > maxweight:
                   maxweight = tvalsum
                   cursol = []
                   start = prenode-1
                   tsign = "e"
            #assert start != None and cursol != None and start != node
            if start!=None and cursol!=None:
               objvals["e"][node,pind] = maxweight
               optsols["e"][node][pind] = optsols[tsign][start][pind] + cursol
    print objvals["e"][nodecount,PARTCOUNT], objvals["d"][nodecount,PARTCOUNT]
    if objvals["e"][nodecount,PARTCOUNT] > objvals["d"][nodecount,PARTCOUNT]:
       return [(start,end) for start,end in optsols["e"][nodecount][PARTCOUNT]], objvals["e"][nodecount,PARTCOUNT]   
    else:
       return [(start,end) for start,end in optsols["d"][nodecount][PARTCOUNT]], objvals["d"][nodecount,PARTCOUNT]     
    #if TESTMODE:    
    #   return [(start,end) for start,end in optsols[nodecount]], objvals[nodecount], [objvals,optsols]
    #else:  
    #   return [(start,end) for start,end in optsols[nodecount]], objvals[nodecount], None


def CRFInfer(weights,nodecount):
    """crf infer
    Args:
       weights,nodecount:
    Returns:   
    """
    objvals = {"e": [0.0] * (nodecount+1), "d": [0.0] * (nodecount+1)}
    optsols = {"e": [[] for node in xrange(nodecount+1)], "d": [[] for node in xrange(nodecount+1)]}
    objvals["e"][1] = weights["neg"][1,1] 
    for node in xrange(2,nodecount+1):
        maxweight, start, cursol,tsign = -1.0e20, None, None, None
        for prenode in xrange(1,node):
            tvalsum = objvals["d"][prenode-1] + weights["pos"][prenode,node]
            if tvalsum >= maxweight:
               maxweight = tvalsum
               cursol = [(prenode,node)]
               start = prenode-1
               tsign = "d"
            tvalsum = objvals["e"][prenode-1] + weights["pos"][prenode,node]       
            if tvalsum >= maxweight:
               maxweight = tvalsum
               cursol = [(prenode,node)]
               start = prenode-1
               tsign = "e"
        #if tsign == "e":
        #   print node
        #   print cursol
        #   print start
        #   print optsols["e"][start]
        #   exit(1)       
        assert start != None and cursol != None
        objvals["d"][node] = maxweight
        optsols["d"][node] = optsols[tsign][start] + cursol
        
        maxweight, start, cursol,tsign = -1.0e20, None, None, None
        for prenode in xrange(1,node+1):
            tvalsum = objvals["d"][prenode-1] + weights["neg"][prenode,node]
            if tvalsum >= maxweight:
               maxweight = tvalsum
               cursol = []
               start = prenode-1
               tsign = "d"
            tvalsum = objvals["e"][prenode-1] + weights["neg"][prenode,node]       
            if tvalsum >= maxweight:
               maxweight = tvalsum
               cursol = []
               start = prenode-1
               tsign = "e"
        assert start != None and cursol != None and start != node
        objvals["e"][node] = maxweight
        optsols["e"][node] = optsols[tsign][start] + cursol
    if objvals["e"][nodecount] >= objvals["d"][nodecount]:
       return [(start,end) for start,end in optsols["e"][nodecount]], objvals["e"][nodecount]   
    else:
       return [(start,end) for start,end in optsols["d"][nodecount]], objvals["d"][nodecount]     
    #if TESTMODE:    
    #   return [(start,end) for start,end in optsols[nodecount]], objvals[nodecount], [objvals,optsols]
    #else:  
    #   return [(start,end) for start,end in optsols[nodecount]], objvals[nodecount], None

    
def MISintervalSingle(node2term,node2notterm,nodecount,infermodel):
    """MIS for single-memm and single-memm2
    Args:
       node2term,node2notterm:
       nodecount,infermodel:
    Returns:
    """
    locs = sorted([node for node in node2term.keys() if node2term[node] - node2notterm[node] > 0.0])
    if infermodel == "single-memm":
       if len(locs) %2 == 1: 
          locs = locs[0:-1]     
       doms = []    
       for ind in xrange(len(locs)/2):
           doms.append((locs[2*ind],locs[2*ind+1])) 
    elif infermodel == "single-memm2":
       doms = []
       start,prenode = locs[0], locs[0]
       domflag = True
       for ind,node in enumerate(locs[1:]):
           if node == prenode+1:
              pass 
           else:
              doms.append((start,prenode))
              start = node
           prenode = node
    return doms        


def MISinterval(weights,nodecount):
    """maximum independent set on interval graph by DP
    Args:
       weights:
       nodecount:
    Returns:
       doms,objval:
       [objvals,optsols]: only in TESTMODE
    """
    objvals = [0.0] * (nodecount+1)
    optsols = [[] for node in xrange(nodecount+1)]
    for node in xrange(2,nodecount+1):
        maxweight, start, cursol = -0.001, None, None
        for prenode in xrange(1,node):
            if objvals[prenode] > maxweight:
               maxweight = objvals[prenode]
               cursol = []
               start = prenode
            if weights[(prenode,node)] > 0.000000001 and objvals[prenode-1] + weights[(prenode,node)] > maxweight: 
               maxweight = objvals[prenode-1] + weights[(prenode,node)]
               cursol = [(prenode,node)]
               start = prenode-1
        assert start != None and cursol != None
        objvals[node] = maxweight
        optsols[node] = optsols[start] + cursol        
    if TESTMODE:    
       return [(start,end) for start,end in optsols[nodecount]], objvals[nodecount], [objvals,optsols]
    else:  
       return [(start,end) for start,end in optsols[nodecount]], objvals[nodecount], None


def checkParam(infermodel,parammodel,domparams):
    """checks params
    """     
    assert infermodel in ["single-memm","single-memm2","crf","pseudo","semicrf","crflatent"]
    if infermodel in ["single-memm","single-memm2"]:
       assert domparams.has_key("term")
    elif infermodel in ["pseudo","crf","semicrf"]:
       assert domparams.has_key("bound") and domparams.has_key("inside") and  domparams.has_key("empty")
    elif infermodel == "crflatent":
       for keystr in domparams.keys():
           assert keystr == "empty" or keystr.startswith("bound") or keystr.startswith("inside")
    if type(domparams.values()[0].values()[0].values()[0]) in [np.ndarray, np.array, list]:
       assert parammodel == "nonparam" 
    return True


def estCRFInfodict(marklist,sortmarkers,nodecounts,width,order):
    """estimates CRF infodict
    Args:
       marklist,sortmarkers,nodecounts,params,width,order:
    Returns:
       infodict:
    """
    infodict = {}
    for mind,markinfo in enumerate(marklist):
        nodecount = nodecounts[mind]  
        mark2pos2count = HistoneUtilities.processMarkData(markinfo)  
        node2count = [None]+[np.array(HistoneUtilities.getCountVec(mark2pos2count,tpos,sortmarkers,width,order),dtype=np.float) for tpos in xrange(1,nodecount+1)]
        infodict[mind] = {"coef": {spos: np.array(node2count[spos]) for spos in xrange(1,nodecount+1)}}
    return infodict

def estPartitionFunc(nodecount,markinfo,params,sortmarkers,width,parammodel,infermodel,order):
    """estimates partiton function
    Args:
       nodecount,markinfo,params,sortmarkers,width,parammodel,infermodel,order:
    Returns:
       logZ: log partition function    
    """
    if infermodel == "crflatent":
       classcount = max([int(keystr.replace("bound","")) for keystr in params.keys() if keystr.find("bound")!=-1])
       paramvec = HistoneUtilities.paramdict2vec(parammodel,params,sortmarkers,width,1,order)
       optparamx = dict2paramvec(paramvec,classcount,"crflatent")
       infodict = estCRFInfodict([markinfo],sortmarkers,[nodecount],width,order)
       logalphadict = estCRFParamsLatent.estLogAlphasCRF(optparamx,[markinfo],infodict,[nodecount],classcount)
       logZ = HistoneUtilities.logSumExp([logalphadict[0][-1]]+[logalphadict[0][-3*cind+1] for cind in xrange(1,classcount+1)])
    elif infermodel == "crf":
       paramvec = HistoneUtilities.paramdict2vec(parammodel,params,sortmarkers,width,1,order)
       optparamx = dict2paramvec(paramvec,1,"crf")
       infodict = estCRFInfodict([markinfo],sortmarkers,[nodecount],width,order)
       logalphadict = estCRFParams.estLogAlphasCRF(optparamx,[markinfo],infodict,[nodecount])
       logZ = HistoneUtilities.logSumExp([logalphadict[0][-2],logalphadict[0][-1]])
    return logZ

def normBernstein(marklist):
    """normalizes data for bernstein nonparametric
    Args:
       marklist: 
    Returns:
       retmarklist:
    """
    mark2max = {}
    for curmarklist in marklist:
        for tmark in curmarklist.keys():
            curcounts = [tcount for tnode,tcount in curmarklist[tmark]]
            mark2max.setdefault(tmark,0.0)
            if max(curcounts) >= mark2max[tmark]:
               mark2max[tmark] = max(curcounts)
    for markval in mark2max.values():
        assert markval <= 11.0
    globmax = 11.0               
    retmarklist = []           
    for curmarklist in marklist:
        putdict = {}
        for tmark in curmarklist.keys():
            putdict[tmark] = list([(tnode,tcount/globmax) for tnode,tcount in curmarklist[tmark]])
        retmarklist.append(putdict)
    return retmarklist

TESTMODE = True
def runner(markinfo,domparams,nodecount,outprefix,domprior,infermodel,prepromodel,parammodel,nooverlap):
    """estimates domain partition
    Args:
       markinfo: marker info dict~(location starts from 1)
       domparams: first and second-order params
       nodecount: n, nodes are from 0 to n-1
       outprefix:
       domprior: domain prior
       infermodel,parmodel,nooverlap: boolean
    Returns:
    """
    assert checkParam(infermodel,parammodel,domparams)
    markinfo = HistoneUtilities.modifyMarkerData([markinfo],[nodecount],prepromodel,False)[0]
    markinfo = normBernstein([markinfo])[0]
    sortmarkers = sorted(markinfo.keys())
    print "used markers are: ",sortmarkers
    markcount,width = len(sortmarkers), len(domparams.values()[0].values()[0].keys())
    order = HistoneUtilities.getOrder(domparams)          
    compcount = 1 if parammodel == "param" else len(domparams.values()[0].values()[0].values()[0])
    stime = time.time()
    
    #fixedval = estPartitionFunc(nodecount,markinfo,domparams,sortmarkers,width,parammodel,infermodel,order)
    fixedval = 0.0
    weights = estWeightsCrf(nodecount,markinfo,domparams,sortmarkers,domprior,width,parammodel,infermodel,order,compcount)
    doms, objval = CRFInfer(weights,nodecount)
    
    etime = time.time()
    respath = "{0}_doms.txt".format(outprefix)
    objoutpath = "{0}_metadata.txt".format(outprefix)
    #assert fixedval >= objval
    #if TESTMODE:
    #   assert TestDomainFinder.testDomEstimate(markinfo,domparams,doms,weights,objval,fixedval,nodecount,parmodel,infermodel,nooverlap,weightside,misside,domprior)
    metadata = {"logobjval": fixedval-objval, "time":etime-stime}
    HistoneUtilities.writeDomainFile(respath,doms)
    HistoneUtilities.writeMetaFile(metadata,objoutpath)
    

def makeParser():
    """
    """
    parser = argparse.ArgumentParser(description='Process domain estimation parameters')
    parser.add_argument('-m', dest='markerpath', type=str, action='store', default='test.marks', help='Marker File(default: test.marks)')
    parser.add_argument('-p', dest='parampath', type=str, action='store', default='params.params', help='Parameter File(default: params.params)')
    parser.add_argument('-o', dest='outprefix', type=str, action='store', default='freq', help='output prefix(default: freq)')
    parser.add_argument('-n', dest='nodecount', type=int, action='store', default=100, help='number of nodes(default: max one in marker data)')
    parser.add_argument('-d', dest='priordist', type=str, action='store', default=None, help='distribution(default: None)')
    parser.add_argument('-a', dest='priorcoef', type=float, action='store', default=0.2, help='prior coef(default: 0.2)')
    parser.add_argument('-l', dest='nooverlap', type=str, action='store', default='True', help='domains must not overlap(default: True)')
    parser.add_argument('-i', dest='infermodel', type=str, action='store', default='crf', help='inference algo(default: crf)')
    parser.add_argument('-t', dest='prepromodel', type=str, action='store', default='loglinear', help='preprocess model(default: loglinear)')
    parser.add_argument('-r', dest='parammodel', type=str, action='store', default='nonparam', help='parameter model(default: nonparam)')
    return parser


if  __name__ =='__main__':
    """runs
    """
    import argparse
    parser = makeParser()
    args = parser.parse_args(sys.argv[1:])
    globals().update(vars(args))
    markinfo = HistoneUtilities.readMarkerFile(markerpath)[0]
    domparams = HistoneUtilities.readDomainParamFile(parampath)
    if priordist in [None,'None']:
       domprior = None
    else:    
       domprior = (priordist,priorcoef)
    runner(markinfo,domparams,nodecount,outprefix,domprior,infermodel,prepromodel,parammodel,nooverlap == 'True')
