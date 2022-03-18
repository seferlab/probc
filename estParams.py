#CRF and SemiCRF parameter estimation
import numpy as np
import scipy as sp
import scipy.io 
import scipy.optimize
import sys
sys.path.append("lib")
import HistoneUtilities
import myutilities as myutil
import gzip
import os
from copy import deepcopy
import itertools
import time
import math
import random
import string
import operator
import sklearn
import sklearn.linear_model
import cPickle as cpickle
#sys.path.append("Tests")
#from TestSEDFMest import TestSEDFMest

def testCrfParamEst(tprobs,nprobs,blen,node2count,inters,sorteddoms,allnodes):
    """tests crf param estimate
    Args:
       tprobs,nprobs,blen:
       node2cout,inter,sorteddoms,allnodes:
    """
    poskeys = tprobs.keys()
    random.shuffle(poskeys)
    for s,e in poskeys[0:50000]:
        sprob = np.array([0.0] * (2*blen))
        sprob[0:blen] = node2count[s] + node2count[e]
        for nind in xrange(s+1,e):
            sprob[blen:] += node2count[nind] 
        assert sum(sprob-tprobs[(s,e)]) < 0.1
    negkeys = nprobs.keys()
    random.shuffle(negkeys)
    for s,e in negkeys[0:50000]:
        sprob = np.array([0.0] * blen)
        for nind in xrange(s,e+1):
            sprob += node2count[nind] 
        assert sum(sprob-nprobs[(s,e)]) < 0.1      
    for start,end in tprobs.keys():
        assert end >= start+1 
    for start,end in nprobs.keys():
        assert end >= start
    allblocks = inters + sorteddoms
    for b1,b2 in itertools.combinations(allblocks,2):
        assert len(set(range(b1[0],b1[1]+1)).intersection(set(range(b2[0],b2[1]+1)))) == 0 
    seennodes = [item for start,end in allblocks for item in xrange(start,end+1)]
    assert len(seennodes) == len(set(seennodes)) and len(seennodes) == len(allnodes) 
    return True


def estCRFParamsNonparam(marklist,domlist,sortmarkers,varcount,nodecounts,params):
    """estimates CRF parameters for nonparam
    Args:
       marklist,domlist,sortmarkers:
       varcount,nodecounts,params:
    Returns:
       infodict,lincoefs:
    """
    basecount = params["basecount"]
    getLog = lambda tval: NEGINF if tval < 1.0e-320 else math.log(tval)
    infodict,loginfodict,lincoefs = {}, {}, np.zeros((varcount,),dtype=np.float64)
    blen,blen2 = varcount/3, 2*varcount/3
    for mind,markinfo in enumerate(marklist):
        sorteddoms = sorted(domlist[mind])
        nodecount = nodecounts[mind]
        mark2pos2count = HistoneUtilities.processMarkData(markinfo)
        prenode2count = [None]+[np.array(HistoneUtilities.getCountVec(mark2pos2count,tpos,sortmarkers,params["width"],1),dtype=np.float) for tpos in xrange(1,nodecount+1)]
        node2count = [None]
        for tnode in xrange(1,nodecount+1):
            putvec = []
            for mval in prenode2count[tnode]:
                usevec = [(1-mval)**(basecount-1)]
                for bind in xrange(1,basecount):
                    usevec.append(usevec[bind-1]*(basecount-bind)*mval/(bind*(1-mval)))
                putvec.extend(usevec)
            node2count.append(np.array(putvec))
        
        infodict[mind] = {"coef": {spos: np.array(node2count[spos]) for spos in xrange(1,nodecount+1)}}
        loginfodict[mind] = {"coef": {spos: np.array([getLog(node2count[spos][varind]) for varind in xrange(blen)]) for spos in xrange(1,nodecount+1)}}
        allnodes = range(1,nodecount+1)
        inters = HistoneUtilities.getEmptyClusters(sorteddoms,allnodes)
        for start,end in sorteddoms:
            sprob = np.array([0.0]* blen2)
            sprob[0:blen] = node2count[start] + node2count[end]
            for tpos in xrange(start+1,end):
                sprob[blen:] += node2count[tpos]
            lincoefs[0:blen2] -= sprob
        for start,end in inters:
            sprob = np.array(node2count[start])
            for tpos in xrange(start+1,end+1):
                sprob += node2count[tpos]
            lincoefs[blen2:] -= sprob
    return infodict,loginfodict,lincoefs


def estLogBetasCRF(paramx,marklist,infodict,nodecounts):
    """estimates betas log for crf: backward
    Args:
       paramx,marklist,infodict,nodecounts:
    Returns:  
       logalphadict: alphadict for markers 
    """
    logbetadict,poslen = {}, len(paramx)/3
    for maind,markinfo in enumerate(marklist):
        nodecount = nodecounts[maind]
        logbetas = np.full((4*(nodecount+1),), NEGINF, dtype=np.float64) #domstart,inside,domend,empty
        logbetas[4*nodecount+2:] = 0.0
        for nind in xrange(nodecount,0,-1):
            boundval,insideval,emptyval = [np.dot(paramx[tind*poslen:(tind+1)*poslen],infodict[maind]["coef"][nind]) for tind in xrange(3)]
            tval1,tval2 = logbetas[4*nind+1] + insideval, logbetas[4*nind+2] + boundval
            logbetas[4*(nind-1)] = HistoneUtilities.logSumExp([tval1,tval2])
            
            tval1,tval2 = logbetas[4*nind+1] + insideval, logbetas[4*nind+2] + boundval
            logbetas[4*(nind-1)+1] = HistoneUtilities.logSumExp([tval1,tval2])
               
            tval1,tval2 = logbetas[4*nind] + boundval, logbetas[4*nind+3] + emptyval
            logbetas[4*(nind-1)+2] = HistoneUtilities.logSumExp([tval1,tval2])
                  
            tval1,tval2 = logbetas[4*nind] + boundval, logbetas[4*nind+3] + emptyval
            logbetas[4*(nind-1)+3] = HistoneUtilities.logSumExp([tval1,tval2])
        logbetadict[maind] = np.array(logbetas)
    return logbetadict


def estLogAlphasCRF(paramx,marklist,infodict,nodecounts):
    """estimates alphas log for crf
    Args:
       paramx,marklist,infodict,nodecounts,maxdomlen:
    Returns:  
       logalphadict: alphadict for markers 
    """
    logalphadict,poslen = {}, len(paramx)/3
    for maind,markinfo in enumerate(marklist):
        nodecount = nodecounts[maind]
        logalphas = np.full((4*(nodecount+1),), NEGINF, dtype=np.float64) #domstart,inside,domend,empty
        logalphas[3] = 0.0
        for nind in xrange(1,nodecount+1):
            boundval,insideval,emptyval = [np.dot(paramx[tind*poslen:(tind+1)*poslen],infodict[maind]["coef"][nind]) for tind in xrange(3)]
            tval1,tval2 = logalphas[4*nind-1] + boundval, logalphas[4*nind-2] + boundval
            logalphas[4*nind] = HistoneUtilities.logSumExp([tval1,tval2])
            
            tval1,tval2 = logalphas[4*nind-4] + insideval, logalphas[4*nind-3] + insideval
            logalphas[4*nind+1] = HistoneUtilities.logSumExp([tval1,tval2])
               
            tval1,tval2 = logalphas[4*nind-4] + boundval, logalphas[4*nind-3] + boundval
            logalphas[4*nind+2] = HistoneUtilities.logSumExp([tval1,tval2])
                  
            tval1,tval2 = logalphas[4*nind-1] + emptyval, logalphas[4*nind-2] + emptyval
            logalphas[4*nind+3] = HistoneUtilities.logSumExp([tval1,tval2])
        logalphadict[maind] = np.array(logalphas)
    return logalphadict

   
def estLogEtasCRF(paramx,logalphadict,marklist,infodict,loginfodict,nodecounts):
    """estimates etas log for crf
    Args:
       paramx,logalphadict,marklist:
       infodict,loginfodict,nodecounts:
    Returns:  
       logetadict:
    """
    logetadict,poslen,poslen2,varcount = {}, len(paramx)/3, 2*len(paramx)/3, len(paramx)
    for maind,markinfo in enumerate(marklist):
        nodecount = nodecounts[maind]
        logalphas = logalphadict[maind]
        logetas = np.full([len(paramx), 4*(nodecount+1)], NEGINF, dtype=np.float64) #domstart,inside,domend,empty
        for nind in xrange(1,nodecount+1):
            boundval,insideval,emptyval = [np.dot(paramx[tind*poslen:(tind+1)*poslen],infodict[maind]["coef"][nind]) for tind in xrange(3)]
            
            for varind in xrange(varcount):   
                tval1,tval3 = logetas[varind,4*nind-1]+boundval, logetas[varind,4*nind-2]+boundval
                tval2 = loginfodict[maind]["coef"][nind][varind] + logalphas[4*nind-1] + boundval if varind < poslen else NEGINF
                tval4 = loginfodict[maind]["coef"][nind][varind] + logalphas[4*nind-2] + boundval if varind < poslen else NEGINF
                logetas[varind,4*nind] = HistoneUtilities.logSumExp([tval1,tval2,tval3,tval4])
                
                tval1,tval3 = logetas[varind,4*nind-4]+insideval, logetas[varind,4*nind-3]+insideval
                tval2 = loginfodict[maind]["coef"][nind][varind-poslen] + logalphas[4*nind-4] + insideval if varind >= poslen and varind < poslen2 else NEGINF
                tval4 = loginfodict[maind]["coef"][nind][varind-poslen] + logalphas[4*nind-3] + insideval if varind >= poslen and varind < poslen2 else NEGINF
                logetas[varind,4*nind+1] = HistoneUtilities.logSumExp([tval1,tval2,tval3,tval4])

                tval1,tval3 = logetas[varind,4*nind-4]+boundval, logetas[varind,4*nind-3]+boundval
                tval2 = loginfodict[maind]["coef"][nind][varind] + logalphas[4*nind-4] + boundval if varind < poslen else NEGINF
                tval4 = loginfodict[maind]["coef"][nind][varind] + logalphas[4*nind-3] + boundval if varind < poslen else NEGINF
                logetas[varind,4*nind+2] = HistoneUtilities.logSumExp([tval1,tval2,tval3,tval4])
                
                tval1,tval3 = logetas[varind,4*nind-1]+emptyval, logetas[varind,4*nind-2]+emptyval
                tval2 = loginfodict[maind]["coef"][nind][varind-poslen2] + logalphas[4*nind-1] + emptyval if varind >= poslen2 else NEGINF
                tval4 = loginfodict[maind]["coef"][nind][varind-poslen2] + logalphas[4*nind-2] + emptyval if varind >= poslen2 else NEGINF
                logetas[varind,4*nind+3] = HistoneUtilities.logSumExp([tval1,tval2,tval3,tval4])    
        logetadict[maind] = np.array(logetas)
    return logetadict


def estJacobianCoefsCRF(logalphadict,logbetadict,nodecounts,markcount,infodict,loginfodict,paramx):
    """estimates jacobian coefs for crf
    Args:
       logalphadict,logbetadict:
       nodecounts,markcount:
       infodict,loginfodict,paramx:
    Returns:
       jacveclist: 
    """
    def estfMdict(markind,varcount,curnodecount):
        """estimates f*M dictionary
        """
        logfMdict = {node: {} for node in xrange(1,curnodecount+1)}
        for node in xrange(1,curnodecount+1):
            boundval,insideval,emptyval = [np.dot(paramx[poslen*tind:poslen*(tind+1)],infodict[markind]["coef"][node]) for tind in xrange(3)]
            logfMdict[node][(0,1)] = [(poslen+ind,item) for ind,item in enumerate(list(loginfodict[markind]["coef"][node] + insideval))] # start to inside
            logfMdict[node][(0,2)] = [(ind,item) for ind,item in enumerate(list(loginfodict[markind]["coef"][node] + boundval))] # start to end
            logfMdict[node][(1,1)] = [(poslen+ind,item) for ind,item in enumerate(list(loginfodict[markind]["coef"][node] + insideval))] # inside to inside
            logfMdict[node][(1,2)] = [(ind,item) for ind,item in enumerate(list(loginfodict[markind]["coef"][node] + boundval))] # inside to empty
            logfMdict[node][(2,0)] = [(ind,item) for ind,item in enumerate(list(loginfodict[markind]["coef"][node] + boundval))] # end to start
            logfMdict[node][(2,3)] = [(poslen2+ind,item) for ind,item in enumerate(list(loginfodict[markind]["coef"][node] + emptyval))] # end to empty
            logfMdict[node][(3,0)] = [(ind,item) for ind,item in enumerate(list(loginfodict[markind]["coef"][node] + boundval))] # empt to start
            logfMdict[node][(3,3)] = [(poslen2+ind,item) for ind,item in enumerate(list(loginfodict[markind]["coef"][node] + emptyval))] # emp to empty
        return logfMdict
    varcount = len(paramx)
    useindices = [(0,1),(0,2),(1,1),(1,2),(2,0),(2,3),(3,0),(3,3)]
    poslen,poslen2 = varcount/3, 2*varcount/3
    jacveclist = []
    for markind in xrange(markcount):
        logfMdict = estfMdict(markind,varcount,nodecounts[markind])
        curlist = {varind:[] for varind in xrange(varcount)}
        albetasum = {tnode:[logalphadict[markind][4*(tnode-1)+ind1]+logbetadict[markind][4*tnode+ind2] for ind1,ind2 in useindices] for tnode in xrange(1,nodecounts[markind]+1)}
        for tnode in xrange(1,nodecounts[markind]+1):
            for etind,(ind1,ind2) in enumerate(useindices):
                for varind,varitem in logfMdict[tnode][(ind1,ind2)]:
                    curlist[varind].append(varitem+albetasum[tnode][etind])
        curarr = [HistoneUtilities.logSumExp(curlist[varind]) for varind in xrange(varcount)]
        jacveclist.append(np.array(curarr))    
    return jacveclist


logalphadict = None
logbetadict = None
normlist = None
def trainCRF(infodict,loginfodict,lincoefs,regcoefs,marklist,domlist,sortmarkers,varcount,nodecounts,params,muvec,initx=None):
    """trains crf
    Args:
       infodict,loginfodict,lincoefs,regcoefs:
       marklist,domlist,sortmarkers:
       varcount,nodecounts,params:
       muvec: group lasso mus
       initx: initial solution
    Returns:
       xvec,objval:
    """
    initx = np.zeros((varcount,),dtype=np.float64) if initx == None else initx
    markcount = len(sortmarkers)
    def jacloglikeCRF(paramx):
        """jacobian of log likelihood
        Args:
           paramx: 
        """
        jacvec = np.array(lincoefs)
        jacveclist = estJacobianCoefsCRF(logalphadict,logbetadict,nodecounts,len(marklist),infodict,loginfodict,paramx)
        if TESTMODE:
           print "here"
           logetadict = estLogEtasCRF(paramx,logalphadict,marklist,infodict,loginfodict,nodecounts)
           testjacvec= jacvec + np.array([sum([math.exp(HistoneUtilities.logSumExp(logetadict[lind][ind,-2:])-normlist[lind]) for lind in xrange(len(marklist))]) for ind in xrange(varcount)])
        jacvec += np.array([sum([math.exp(jacveclist[markind][varind]-normlist[markind]) for markind in xrange(len(marklist))]) for varind in xrange(varcount)])
        jacvec += (2.0*params["grlambda"]) * paramx
        #jacvec += (2.0*params["lambda"]) * np.array([item for uind in xrange(3) for mind in xrange(markcount) for item in list(np.dot(regcoefs,paramx[(markcount*uind+mind)*params["width"]*params["basecount"]:(markcount*uind+mind+1)*params["width"]*params["basecount"]]))])
        #jacvec += 2.0*params["lambda"]*np.array([paramx[(markcount*uind+mind)*params["width"]*params["basecount"]:(markcount*uind+mind+1)*params["width"]*params["basecount"]]/float(muvec[uind,mind]) for uind in xrange(3) for mind in xrange(markcount)]).flatten()
        if TESTMODE:
           assert logalphadict != None and logbetadict != None and normlist != None
        #   print "iter info: ",np.linalg.norm(jacvec-tjacvec) 
        #   assert np.linalg.norm(jacvec-tjacvec) < 0.001  
        return jacvec
    def loglikeCRF(paramx):
        """log likelihood 
        Args:
           paramx:
        Returns:
           objval:
        """
        global logalphadict,logbetadict,normlist
        tobjval = np.dot(paramx,lincoefs)
        logalphadict = estLogAlphasCRF(paramx,marklist,infodict,nodecounts)
        logbetadict = estLogBetasCRF(paramx,marklist,infodict,nodecounts)
        normlist = [HistoneUtilities.logSumExp([logalphadict[tind][-2],logalphadict[tind][-1]]) for tind in xrange(len(marklist))]
        tobjval += sum(normlist)
        tobjval += params["grlambda"]*(np.linalg.norm(paramx)**2)
        #sideval = 0.0
        #for uind in xrange(3):
        #    for mind in xrange(markcount):
        #        usex = paramx[(markcount*uind+mind)*params["width"]*params["basecount"]:(markcount*uind+mind+1)*params["width"]*params["basecount"]]
        #        sideval += (np.linalg.norm(usex)**2)/muvec[uind,mind]
        #        sideval += np.dot(np.dot(usex,regcoefs),usex)
        #tobjval += params["lambda"]*sideval
        print "current obj: ",tobjval
        if TESTMODE:
           for item in normlist:
               assert item != 0.0   
           assert tobjval >= 0.0
        return tobjval
    xvec,objval,d = scipy.optimize.fmin_l_bfgs_b(loglikeCRF, initx, fprime=jacloglikeCRF,maxiter=params['itercount'])
    #xvec,objval,d = scipy.optimize.fmin_l_bfgs_b(loglikeCRF, initx, approx_grad=1, disp=None, maxiter=params['itercount'],epsilon=1e-08)
    print ",".join([str(item) for item in xvec])
    print "objval: ",objval
    return xvec,objval

def loglikeEst(paramx,lincoefs,marklist,infodict,nodecounts,regcoefs,sortmarkers,params,tmuvec):
    """negative log likelihood obj + penalty
    Args:
       paramx,lincoefs,marklist,infodict,nodecounts:
       regcoefs,sortmarkers,params,tmuvec:
    Returns:
       tobjval:   
    """
    tobjval = np.dot(paramx,lincoefs)
    hlogalphadict = estLogAlphasCRF(paramx,marklist,infodict,nodecounts)
    hlogbetadict = estLogBetasCRF(paramx,marklist,infodict,nodecounts)
    hnormlist = [HistoneUtilities.logSumExp([hlogalphadict[tind][-2],hlogalphadict[tind][-1]]) for tind in xrange(len(marklist))]
    markcount = len(sortmarkers)
    tobjval += sum(hnormlist)
    tobjval += params["grlambda"]*(np.linalg.norm(paramx)**2)
    #sideval = 0.0
    #for uind in xrange(3):
    #    for mind in xrange(markcount):
    #        usex = paramx[(markcount*uind+mind)*params["width"]*params["basecount"]:(markcount*uind+mind+1)*params["width"]*params["basecount"]]
    #        sideval += (np.linalg.norm(usex)**2)/tmuvec[uind,mind]
    #        sideval += np.dot(np.dot(usex,regcoefs),usex)
    #tobjval += params["lambda"]*sideval
    return tobjval
        
def iterativeRunner(marklist,domlist,sortmarkers,varcount,nodecounts,params):
    """iterative runner
    Args:
       marklist,domlist:
       sortmarkers:
       varcount,nodecounts:
       params:
    Returns:
    """
    muvec = np.zeros((3,len(sortmarkers)),dtype=np.float)
    for tind in xrange(3):
        muvec[tind,:] = 1.0/len(sortmarkers)
    regcoefs = estBernRegCoefs(params["basecount"]-1)
    assert testestReg(params['basecount'])     
    solx = np.zeros((varcount,),dtype=np.float64)
    infodict,loginfodict,lincoefs = estCRFParamsNonparam(marklist,domlist,sortmarkers,varcount,nodecounts,params)
    solobjval = loglikeEst(solx,lincoefs,marklist,infodict,nodecounts,regcoefs,sortmarkers,params,muvec) 
    ind = 0
    while True:
        print ind,solobjval
        cursolx,curobjval = trainCRF(infodict,loginfodict,lincoefs,regcoefs,marklist,domlist,sortmarkers,varcount,nodecounts,params,muvec,initx=solx)
        #if ind == 0 and curobjval > solobjval:
        #   break
        ind += 1
        #if ind == 1:
        #   solx = np.array(cursolx)
        #   solobjval = curobjval
        #   break
        curobjval -= estPenaltyNonparam(cursolx,muvec,len(sortmarkers),params["basecount"],params["width"],params["lambda"])
        assert curobjval >= 0.0
        muvec = estMuVecNonparam(cursolx,sortmarkers,varcount,params["basecount"],params["width"])
        curobjval += estPenaltyNonparam(cursolx,muvec,len(sortmarkers),params["basecount"],params["width"],params["lambda"])
        testtobjval = loglikeEst(cursolx,lincoefs,marklist,infodict,nodecounts,regcoefs,sortmarkers,params,muvec)
        assert abs(testtobjval - curobjval) < 0.1
        if curobjval >= solobjval - 0.01:
           break
        solobjval,solx = curobjval, np.array(cursolx)
    return solx,solobjval,muvec
     
def estPenaltyNonparam(tx,tmuvec,markcount,compcount,width,curlam):
    """estimates penalty part of objective
    Args:
       tx,tmuvec,markcount:
       compcount,width,curlam: smoothing parameter
    Returns: 
    """
    sideval = 0.0
    for uind in xrange(3):
        for mind in xrange(markcount):
            usex = tx[(markcount*uind+mind)*width*compcount:(markcount*uind+mind+1)*width*basecount]
            sideval += (np.linalg.norm(usex)**2)/tmuvec[uind,mind]
    return curlam*sideval

def estMuVecNonparam(solx,sortmarkers,varcount,basecount,width):
    """estimates muvec for nonparametric case
    Args:
       solx,sortmarkers:
       varcount,basecount,width:
    Returns:
       tmuvec:
    """
    tmuvec = np.zeros((3,len(sortmarkers)),dtype=np.float)
    blocklen = varcount/3
    for tind in xrange(3):
        tmuvec[tind,:] = [np.linalg.norm(solx[(tind*blocklen)+(mind*basecount*width):(tind*blocklen)+((mind+1)*basecount*width)]) for mind in xrange(len(sortmarkers))]
        tmuvec[tind,:] = tmuvec[tind,:] / float(sum(tmuvec[tind,:]))
    return tmuvec 

def testestReg(basecount):
    """tests regularization estimation
    Args:
       basecount:
    """
    basecount = 3
    xvec = [random.uniform(0,1) for tind in xrange(basecount+1)]
    regcoefs = estBernRegCoefs(basecount)  
    for node1 in xrange(basecount+1):
        for node2 in xrange(basecount+1):
            assert abs(regcoefs[node1,node2] - regcoefs[node2,node1]) < 0.00001
    eigs = scipy.linalg.eigh(regcoefs)[0]
    for eigval in eigs:
        assert eigval >= -0.000000001            
    return True
        
def estBernRegCoefs(basecount):
    """estimates bernstein regularization coefs
    Args:
       basecount:
    Returns:
       regcoefs: array of coefs   
    """
    def combval(n,r):
        return math.factorial(n)/(math.factorial(r)*math.factorial(n-r))
    regcoefs = np.zeros((basecount+1,basecount+1),dtype=np.float64)
    for i in xrange(basecount+1):
        for j in xrange(basecount+1):
            for q in xrange(max(0,i-basecount+2),min(2,i)+1):
                for r in xrange(max(0,j-basecount+2),min(2,j)+1):
                    regcoefs[i,j] += math.pow(-1,q+r) * combval(2,q) * combval(2,r) * combval(basecount-2,i-q) * combval(basecount-2,j-r) * scipy.special.beta(i+j-q-r+1,2*basecount-3-i-j+q+r)
    return regcoefs * (basecount*(basecount-1))**2
        
def getVarcount(markcount,width,basecount):
    """gets var count
    Args:
       markcount,width,basecount:
    Returns:
       varcount:
    """
    varcount = 3 * basecount * width * markcount
    return varcount

def checkParamsNonParam(params):
    """checks params of nonparametric case
    Args:
       params:
    Returns:    
    """
    assert params['prepromodel'] in ["linear","loglinear","binary","binary0.5","colnorm","poisson0.9","poisson0.99"] and params['width']>=1 and params['itercount']>=4
    return True
    
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
    
TESTMODE = False
NEGINF = -1.0e50
def runner(marklist,domainlist,nodecounts,outprefix,params):
    """estimates SEDFM parameters
       Assumes both domains and marker indices start from 1 not 0
    Args:
       marklist: marker list
       domainlist: list of domains(start from 1 not 0)
       nodecounts: nodecounts of all domains
       outprefix:
       params: model order,lambda
    Returns:
    """
    assert checkParamsNonParam(params)
    marklist = HistoneUtilities.modifyMarkerData(marklist,nodecounts,params['prepromodel'],False)
    #marklist = normBernstein(marklist)
    sortmarkers = sorted(list(set(mark for markinfo in marklist for mark in markinfo.keys())))
    print "input markers are: ",sortmarkers
    markcount = len(sortmarkers)
    stime = time.time()
    varcount = getVarcount(markcount,params['width'],params['basecount'])
    respath = "{0}_domparams.txt".format(outprefix)
    objoutpath = "{0}_dommetadata.txt".format(outprefix)
    solx,objval,muvec = iterativeRunner(marklist,domlist,sortmarkers,varcount,nodecounts,params)
    paramdict=HistoneUtilities.sol2dict([solx[0:varcount/3],solx[varcount/3:2*varcount/3],solx[2*varcount/3:]],sortmarkers,params['width'],"crf",None,params["basecount"])   
    etime = time.time()
    metadata = {"time":etime-stime, "logobjval": -1.0*objval}
    if False: 
       if params['model'] in ["linear","binary"] and params["order"] == 1:
          sidecoefs = [Xdom,ydom,solx] 
       elif params['model'] in ["linear","binary"]:
          sidecoefs = [Xdom,ydom,lincoefs,logcoefs,solx,objval,muvec]    
       elif params['model'] == "nonparam":
          sidecoefs = [lincoefs,logcoefs,solx,objval,muvec,COMPCOUNT]       
       assert TestSEDFMest.testDomainParamEstimateVomm(marklist,domlist,paramdict,sortmarkers,sidecoefs,nodecounts,params,varcount) 
    HistoneUtilities.writeDomainParamFile(respath,paramdict)
    HistoneUtilities.writeMetaFile(metadata,objoutpath)

        
def makeParser():
    """
    """
    parser = argparse.ArgumentParser(description='Parameter estimation')
    parser.add_argument('-m', dest='markerpath', type=str, action='store', default='train.marklist', help='Marker File(default: train.marklist)')
    parser.add_argument('-p', dest='domainpath', type=str, action='store', default='train.domainlist', help='List of domains File(default: train.domainlist)')
    parser.add_argument('-o', dest='outprefix', type=str, action='store', default='freq', help='output prefix(default: freq)')
    parser.add_argument('-l1', dest='lambdaval', type=float, action='store', default=0.0, help = 'smoothness parameter')
    parser.add_argument('-l2', dest='grlambdaval', type=float, action='store', default=1.0, help = 'lambda for coefficient sparsity')
    parser.add_argument('-w', dest='width', type=int, action='store', default=1, help='effect width(default: 1)')
    parser.add_argument('-t', dest='prepromodel', type=str, action='store', default='loglinear', help='preprocess model(default: loglinear)')
    parser.add_argument('-c', dest='itercount', type=int, action='store', default=1000, help='iteration count(default: 1000)')
    parser.add_argument('-k', dest='basecount', type=int, action='store', default=1, help='# of base kernels(default: 1)')
    parser.add_argument('-cb', dest='cb', type=float, action='store', default=1.0, help='relative weight of boundary(default: 1.0)')
    parser.add_argument('-ci', dest='ci', type=float, action='store', default=1.0, help='relative weight of interior(default: 1.0)')
    parser.add_argument('-ce', dest='ce', type=float, action='store', default=1.0, help='relative weight of external(default: 1.0)')
    return parser


if  __name__ =='__main__':
    """runs
    """
    import argparse
    parser = makeParser()
    args = parser.parse_args(sys.argv[1:])
    globals().update(vars(args))
    marklist = HistoneUtilities.readMarkerFile(markerpath)
    domlist,nodecounts = HistoneUtilities.readMultiDomainFile(domainpath)
    params = {'lambda':lambdaval,'grlambda':grlambdaval,'itercount':itercount, "width":width, "prepromodel":prepromodel,'basecount':basecount,'ci':ci,'ce':ce,'cb':cb}
    runner(marklist,domlist,nodecounts,outprefix,params)

    
