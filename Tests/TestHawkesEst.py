#SEDFM parameter estimation Tests
import networkx as nx
import numpy as np
import math
import random
import os
import sys
sys.path.append("lib")
import HistoneUtilities
import itertools
from copy import deepcopy


class TestSEDFMest():

    @staticmethod
    def estParamEstObj(marklist,domlist,paramdict,sortmarkers,logcoefs,lincoefs,nodecount):
        """estimate objective for given paramdict
        Args:
           marklist:
           domlist:
           paramdict:
           sortmarkers:
           logcoefs:
           lincoefs:
           nodecount:
        Returns:
           objval:
        """
        objval = 0.0
        paramvec = HistoneUtilities.paramdict2vec(paramdict,sortmarkers)
        for mainind,markinfo in enumerate(marklist):
            doms = domlist[mainind]
            sorteddoms = sorted(doms)
            boundlocs,indomlocs,remlocs = HistoneUtilities.getLocs(sorteddoms,nodecount)
            mark2pos2count = HistoneUtilities.processMarkData(markinfo)
            for node in boundlocs:
                countvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers),dtype=np.float)
                objval += math.log(1.0 + math.exp(np.dot(countvec,paramvec)))
                objval += np.dot(countvec*-1.0, paramvec)
            for node in indomlocs | remlocs:
                countvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers),dtype=np.float)
                objval += math.log(1.0 + math.exp(np.dot(countvec,paramvec)))
        return objval

    @staticmethod
    def checkSingleMark(marklist):
        """checks whether there is single document
        Args:
           marklist:
        Returns:
           bool:
        """
        for markinfo in marklist:
            mark2pos2count = HistoneUtilities.processMarkData(markinfo)
            for mark in mark2pos2count.keys():
                for val in mark2pos2count[mark].values():
                    if val > 2:
                       return False  
        return True

    @staticmethod
    def testInputData(mark2sortedlist):
        """tests input data
        Args:
           mark2sortedlist:
        Returns:
           bool: 
        """
        for mainind,mark2sortinfo in enumerate(mark2sortedlist):
            for ind1,(pos1,mark1,count1) in enumerate(mark2sortinfo):
                assert pos1 < nodecount
                for ind2,(pos2,mark2,count2) in xrange(len(mark2sortinfo)):
                    assert pos2 <= pos
        return True

    @staticmethod
    def checkyvec(yvec,markcount):
        """checks y vec sum to 1
        Args:
           yvec:
           markcount:
        Returns:
           bool: true / false   
        """
        for ind1 in xrange(markcount):
            cursum = sum([yvec[markcount*ind1+ind2] for ind2 in xrange(markcount)])
            assert abs(cursum-1.0) < 0.0001
        return True
    
    @staticmethod
    def testHistoneParamEstimate(marklist,mark2sortedlist,mudict,alphadict,yvec,sortedmarkers,nodecount,minobjval):
        """tests hawkes histone param estimate
        Args:
           marklist:
           mark2sortedlist:
           mudict,alphadict:
           yvec:
           sortedmarkers:
           nodecount:
           minobjval:
        Returns:
           bool: true or false
        """
        markcount = len(sortedmarkers)
        assert checkyvec(yvec,markcount)
        assert testInputData(mark2sortedlist)
        return True
        #estobjval = TestSEDFMest.estParamEstObj(marklist,domlist,paramdict,sortmarkers,logcoefs,lincoefs,nodecount)
        #estobjval += sum([params['lambda1']*abs(item) for item in solx])
        #assert abs(estobjval - minobjval) <= 0.1
        #return True

  

