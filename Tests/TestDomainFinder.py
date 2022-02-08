#Domain Finder Tests
import networkx as nx
import numpy as np
#import scipy as sp
import math
import random
import os
import sys
import itertools
#import HistoneUtilities
from copy import deepcopy


class TestDomainFinder():

    @staticmethod
    def testMISDP(misside,nodecount,weights):
        """tests dynamic progragmming for MIS
        Args:
           misside
           nodecount,weights:
        Returns:
           bool:
        """
        [objvals,optsols] = misside
        assert objvals[0] == 0
        for mind in xrange(1,len(objvals)):
            for ind in xrange(mind):
                assert objvals[mind] - objvals[ind] >= -1.0 * 1e-6
        for mind in xrange(1,len(optsols)):
            myobjval = sum([0.0]+[weights[(start,end)] for start,end in optsols[mind]])
            assert abs(myobjval - objvals[mind]) < 1e-4  
            TestDomainFinder.checkOverlap(optsols[mind])
        for objval in objvals:
            assert objval >= 0.0
        for node in xrange(len(optsols)):
            curobjval = sum(weights[(start,end)] for start,end in optsols[node])
            assert abs(curobjval - objvals[node]) < 0.0001
        intersect = lambda (s1,e1), (s2,e2): False if (e1 < s2 or e2 < s1) else True    
        for ind in xrange(0):
            for domlen in xrange(1,nodecount/2):
                randkeys = list(weights.keys())
                random.shuffle(randkeys)
                itersol = []
                for start,end in randkeys:
                    if len(itersol) == domlen:
                       break 
                    flag = False
                    for curstart,curend in itersol:
                        if intersect((curstart,curend),(start,end)):
                           flag = True
                           break
                    if not flag:
                       itersol.append((start,end))    
                curobjval = sum(weights[(start,end)] for start,end in itersol)
                assert curobjval <= objvals[nodecount]
        return True       
               
    @staticmethod
    def testWeightProbs(weightside,nodecount,infermodel):
        """tests weight probs
        Args:
           weightside,nodecount:
        """
        if infermodel == "double":
           [node2start,node2end,node2notstart,node2notend,cumnode2start,cumnode2end,cumnode2notstart,cumnode2notend] = weightside
           blockset1 = [node2start,node2end,node2notstart,node2notend]
           blockset2 = [(node2start,node2notstart), (node2end,node2notend)]
           blockset3 = [(node2start,cumnode2start),(node2end,cumnode2end),(node2notstart,cumnode2notstart),(node2notend,cumnode2notend)]
        elif infermodel == "single-memm":
           [node2term,node2notterm,cumnode2term,cumnode2notterm] = weightside
           blockset1 = [node2term,node2notterm]   
           blockset2 = [(node2term,node2notterm)]
           blockset3 = [(node2term,cumnode2term),(node2notterm,cumnode2notterm)]
        for block in blockset1:
            assert not block.has_key(0) and block.has_key(nodecount) and not block.has_key(nodecount+1)
            for node in block.keys():
                assert math.exp(block[node]) >= -0.00001 and math.exp(block[node]) < 1.00001
        for block1,block2 in blockset2:
            for node in block1.keys():
                assert abs(math.exp(block1[node]) + math.exp(block2[node]) -1) < 0.0001
        for block,cumblock in blockset3:
            assert cumblock[0] == 0.0       
            #for node in xrange(1,nodecount+1):
            #    assert abs(cumblock[node] - cumblock[node-1] - block[node]) < 100.0 #large values
        return True
       
    @staticmethod
    def compareObj(doms,nodecount,objval,fixedval,weightside,weights,domprior):
        """compares objective
        Args:
           doms:
           nodecount:
           objval,fixedval:
           weightside,weights:
        Returns:
           bool:
        """
        [node2start,node2end,node2notstart,node2notend,cumnode2start,cumnode2end,cumnode2notstart,cumnode2notend] = weightside 
        remlocs = set(range(1,nodecount+1)).difference(set([loc for start,end in doms for loc in xrange(start,end+1)]))       
        logobj = 2.0 * sum(node2notstart[tloc] for tloc in remlocs)
        logobj += sum(node2start[tloc] for tlocset in doms for tloc in tlocset)
        logobj += sum(node2end[tloc] for tlocset in doms for tloc in tlocset)
        logobj += 2.0*sum(node2notend[tloc] for start,end in doms for tloc in xrange(start+1,end))
        if domprior not in [None, 'None']:
           priorfunc, priorcoef = domprior
           if priorfunc == "geometric":
              prifunc = lambda domlen,coef: (domlen-1)*math.log(1.0-coef) + math.log(coef)
           elif priorfunc == "powerlaw": #zips's kind
              prifunc = lambda domlen,coef: math.log(max(0.0000000000000001,math.pow(domlen,-1.0*coef) - math.pow(domlen+1,-1.0*coef)))  
           logobj += sum(prifunc(end-start+1,priorcoef) for start,end in doms)
        assert abs(logobj - objval - fixedval) < 100.0
        assert abs(objval - sum(weights[(start,end)] for start,end in doms)) <= 0.0001
        return True

    @staticmethod 
    def testDomEstimate(markinfo,domparams,doms,weights,objval,fixedval,nodecount,model,infermodel,nooverlap,weightside,misside,domprior):
        """tests domain estimate
        Args:
           markinfo:
           domparams:
           doms:
           weights:
           objval,fixedval:
           nodecount,model,infermodel:
           nooverlap,weightside,misside:
        Returns:
           bool: true or false
        """ 
        print "avg domain length: ",np.mean([end-start+1 for start,end in doms])
        seennodes = [ind for start,end in doms for ind in xrange(start,end+1)]
        print "inter-domain nodes: ",set(range(1,nodecount+1)).difference(set(seennodes))
        print "node info: ",len(seennodes), nodecount  
        neg,pos = len([val for val in weights.values() if val < 0.0]), len([val for val in weights.values() if val >= 0.0])   
        print pos, neg 
        assert TestDomainFinder.testWeightProbs(weightside,nodecount,infermodel)
        if nooverlap:
           assert TestDomainFinder.testMISDP(misside,nodecount,weights)
           assert TestDomainFinder.checkOverlap(doms)
           assert TestDomainFinder.compareObj(doms,nodecount,objval,fixedval,weightside,weights,domprior)
        boundlocs = [loc for dom in doms for loc in dom]
        assert min(boundlocs) >= 1 and max(boundlocs) <= nodecount
        return True

    @staticmethod
    def checkOverlap(doms):
        """check overlap
        Args:
           doms: domains
        Returns:
           bool: true or false
        """
        intersect = lambda (s1,e1), (s2,e2): False if (e1 < s2 or e2 < s1) else True
        for dom1,dom2 in itertools.combinations(doms,2):
            assert not intersect(dom1,dom2)
        return True
   
  

