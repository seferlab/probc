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
    def estNonParamObj(paramdict,marklist,domlist,sortmarkers,params,nodecount,compcount,stmuvec,endmuvec):
        """
        """
        def estParam2Vec(keyw):
            """
            """
            xvec = [paramdict[keyw][mark][tw][cind] for markind,mark in enumerate(sortmarkers) for tw in xrange(1,params['width']+1) for cind in xrange(compcount)]
            if params['order'] == 2:
               xvec += [paramdict[keyw][(mark1,mark2)][tw][cind] for mark1,mark2 in itertools.combinations(sortmarkers,2) for tw in xrange(1,params['width']+1) for cind in xrange(compcount)]
            return xvec
        cobjval = 0.0
        stx, endx = estParam2Vec("start"), estParam2Vec("end") 
        for mainind,markinfo in enumerate(marklist):
            sorteddoms = sorted(domlist[mainind])
            boundlocs,indomlocs,remlocs = HistoneUtilities.getLocs(sorteddoms,nodecount)
            mark2pos2count = HistoneUtilities.processMarkData(markinfo)
            for node in boundlocs:
                retvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,params["width"],params['order']),dtype=np.float)  
                countvec = HistoneUtilities.getCompPars(retvec,compcount) 
                tval1, tval2 = np.dot(countvec,stx), np.dot(countvec,endx)
                cobjval += math.log(math.exp(tval1)+1.0) + math.log(math.exp(tval2)+1.0) - tval1 - tval2
            for node in indomlocs:
                retvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,params["width"],params['order']),dtype=np.float)  
                countvec = HistoneUtilities.getCompPars(retvec,compcount)
                tval = np.dot(countvec,endx)
                cobjval += 2*math.log(math.exp(tval)+1.0)
            for node in remlocs:
                retvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,params["width"],params['order']),dtype=np.float)  
                countvec = HistoneUtilities.getCompPars(retvec,compcount)  
                tval = np.dot(countvec,stx)
                cobjval += 2*math.log(math.exp(tval)+1.0)
        for vecx,vecmuvec in [(stx,stmuvec),(endx,endmuvec)]:                
            cobjval += params['lambda'] * sum([(np.linalg.norm(vecx[ind*params['width']*compcount:(ind+1)*params['width']*compcount])**2)/vecmuvec[ind] for ind in xrange(len(vecx)/(2*compcount*params['width']))])           
        return cobjval


    @staticmethod
    def estParamEstObj(marklist,domlist,paramdict,sortmarkers,nodecounts,params):
        """estimate objective for given paramdict
        Args:
           marklist,domlist,paramdict:
           sortmarkers,nodecounts,params:
        Returns:
           objval:
        """    
        def estDoubleBound():  
            tval1, tval2 = np.dot(countvec,stx), np.dot(countvec,endx)
            return (math.log(math.exp(tval1)+1.0) + math.log(math.exp(tval2)+1.0) - tval1 - tval2)
        def estDoubleIndom():  
            tval = np.dot(countvec,endx)
            return 2.0*math.log(math.exp(tval)+1.0)
        def estDoubleRem():  
            tval = np.dot(countvec,stx)
            return 2.0*math.log(math.exp(tval)+1.0)  
        def estSinglememmBound(): 
            tval = np.dot(countvec,termx)
            return (math.log(math.exp(tval)+1.0) - tval) 
        def estSinglememmIndom():  
            tval = np.dot(countvec,termx)
            return math.log(math.exp(tval)+1.0) 
        def estSinglememmRem():  
            tval = np.dot(countvec,termx)
            return math.log(math.exp(tval)+1.0)
        def estSinglememm2Bound(): 
            tval = np.dot(countvec,termx)
            return (math.log(math.exp(tval)+1.0) - tval) 
        def estSinglememm2Indom():  
            tval = np.dot(countvec,termx)
            return (math.log(math.exp(tval)+1.0) - tval) 
        def estSinglememm2Rem():  
            tval = np.dot(countvec,termx)
            return math.log(math.exp(tval)+1.0)    
        def estParam2Vec(keyw):
            xvec = [paramdict[keyw][mark][tw] for markind,mark in enumerate(sortmarkers) for tw in xrange(1,params['width']+1)]
            if params['order'] == 2:
               xvec += [paramdict[keyw][(mark1,mark2)][tw] for mark1,mark2 in itertools.combinations(sortmarkers,2) for tw in xrange(1,params['width']+1)]
            return xvec

        if params["infermodel"] in ["single-memm","single-memm2"]:   
           termx = estParam2Vec("term")
        elif params["infermodel"] == "double":    
           stx, endx = estParam2Vec("start"), estParam2Vec("end")   
        boundfunc = "est{0}Bound".format(params["infermodel"].replace("-","").capitalize())
        indomfunc = "est{0}Indom".format(params["infermodel"].replace("-","").capitalize()) 
        remfunc = "est{0}Rem".format(params["infermodel"].replace("-","").capitalize())         
        objval = 0.0 
        for mainind,markinfo in enumerate(marklist):
            sorteddoms = sorted(domlist[mainind])
            boundlocs,indomlocs,remlocs = HistoneUtilities.getLocs(sorteddoms,nodecounts[mainind])
            mark2pos2count = HistoneUtilities.processMarkData(markinfo)
            for node in boundlocs:
                countvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,params["width"],params['order']),dtype=np.float) 
                objval += locals()[boundfunc]()      
            for node in indomlocs:
                countvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,params["width"],params['order']),dtype=np.float)      
                objval += locals()[indomfunc]()   
            for node in remlocs:
                countvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,params["width"],params['order']),dtype=np.float)      
                objval += locals()[remfunc]()  
        if params["infermodel"] in ["single-memm","single-memm2"]:          
           objval += params['lambda'] * sum([abs(item) for item in termx])   
        elif params["infermodel"] == "double":            
           objval += params['lambda'] * sum([abs(item) for item in stx] + [abs(item) for item in endx])            
        return objval

    @staticmethod
    def testDomainParamInput(marklist,domlist,nodecounts):
        """tests domain param input
        Args:
           marklist,domlist:
           nodecount:
        Returns:
        """    
        assert len(marklist) == len(domlist)
        for ind,doms in enumerate(domlist):
            boundlocs = [loc for dom in doms for loc in dom]
            assert len(boundlocs) == len(set(boundlocs))
            assert min(boundlocs) >= 1 and max(boundlocs) <= nodecounts[ind]
        return True
          
    @staticmethod
    def testPreParamDataVomm(marklist,domlist,sortmarkers,nodecounts,params):
        """test preprocesed data for vomm
        """
        assert TestSEDFMest.testDomainParamInput(marklist,domlist,nodecounts)
        for mainind,markinfo in enumerate(marklist):
            sorteddoms = sorted(domlist[mainind]) 
            boundlocs,indomlocs,remlocs = HistoneUtilities.getLocs(sorteddoms,nodecounts[mainind])
            assert len(boundlocs & remlocs) == 0 and len(boundlocs & indomlocs) == 0 and len(remlocs & indomlocs) == 0
            alllocs = boundlocs | indomlocs | remlocs
            allnodes = range(1,nodecounts[mainind]+1)
            assert len(alllocs ^ set(allnodes)) == 0 
            inters = HistoneUtilities.getEmptyClusters(sorteddoms,allnodes)
            seennodes = set(node for start,end in inters for node in range(start,end+1))
            seennodes2 = set(node for start,end in sorteddoms for node in range(start,end+1))
            assert len(set(range(1,nodecounts[mainind]+1)) ^ set(seennodes | seennodes2)) == 0
            assert len(set(seennodes).intersection(seennodes2)) == 0 
            mark2pos2count = HistoneUtilities.processMarkData(markinfo)
            for tnode in list(seennodes) + list(seennodes2):
                countvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,tnode,sortmarkers,params["width"],params['order']),dtype=np.float)
                assert TestSEDFMest.testCountVec(tnode,countvec,params['width'],sortmarkers,mark2pos2count)
        return True
       
    @staticmethod
    def testParamDict(mark2pos2count,model):
        """test param dict 
        """
        if model == "binary":
           for marker in mark2pos2count.keys():
              for pos in mark2pos2count[marker].keys():
                  assert mark2pos2count[marker][pos] in [0.0,1.0]   
        return True

    @staticmethod
    def testPreNonParamData(marklist,domlist,sortmarkers,varcount,nodecount,params,compcount,logcoefs):
        """tests preprocessed data
        """  
        def isSame(arr1,arr2):
            """
            """
            if len(arr1) != len(arr2):
               return False 
            for ind in xrange(len(arr1)):
                if (arr1[ind] - arr2[ind]) > 0.00001:
                   return False
            return True       
        for mainind,markinfo in enumerate(marklist):
            sorteddoms = sorted(domlist[mainind])
            boundlocs,indomlocs,remlocs = HistoneUtilities.getLocs(sorteddoms,nodecount)
            mark2pos2count = HistoneUtilities.processMarkData(markinfo)
            assert TestSEDFMest.testParamDict(mark2pos2count,params["model"])
            stcurlogcoefs,endcurlogcoefs = stlogcoefs[mainind], endlogcoefs[mainind]
            stind,endind = 0,0
            for node in list(boundlocs) + list(indomlocs) + list(remlocs):
                retvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,params["width"],params['order']),dtype=np.float)
                assert TestSEDFMest.testCountVec(node,retvec,params['width'],sortmarkers,mark2pos2count)
                countvec = HistoneUtilities.getCompPars(retvec,compcount) 
                assert TestSEDFMest.testNonParamCoefs(countvec,compcount)
                if node in boundlocs:
                   assert isSame(stcurlogcoefs[stind],countvec) and isSame(endcurlogcoefs[endind],countvec)
                   stind += 1
                   endind += 1
                elif node in indomlocs:
                   assert isSame(endcurlogcoefs[endind],countvec) and isSame(endcurlogcoefs[endind+1],countvec)
                   endind += 2
                elif node in remlocs:
                   assert isSame(stcurlogcoefs[stind],countvec) and isSame(stcurlogcoefs[stind+1],countvec)
                   stind += 2  
        return True 

    @staticmethod
    def testNonParamCoefs(countvec,compcount):
        """tests nonparametric coefs
        Args:
            countvec,compcount:
        Returns:
        """     
        blockcount = len(countvec)/compcount        
        for tind in xrange(blockcount):
            vals = countvec[tind*compcount:(tind+1)*compcount]
            if vals[0] == 0.0:
               for val in vals:
                   assert val == 0.0
            else:
               ratios = [vals[ind]/float(vals[ind-1]) for ind in xrange(1,compcount)]  
               for r1,r2 in itertools.combinations(ratios,2):
                   assert abs(r1-r2) < 0.001    
        return True

    @staticmethod
    def testCountVec(node,countvec,width,sortmarkers,mark2pos2count):
        """tests count vec 
        Args:
           node,countvec,width:
           sortmarkers,mark2pos2count:
        Returns:
        """
        linvec,quadvec = countvec[0:width*len(sortmarkers)], countvec[width*len(sortmarkers):]
        for tind,item in enumerate(linvec):
            markind,remwidth = tind / width, tind % width
            if remwidth == 0:
               val = mark2pos2count[sortmarkers[markind]].get(node,0.0)
               assert val == item  
            else: 
               val = mark2pos2count[sortmarkers[markind]].get(node-remwidth,0.0)
               val += mark2pos2count[sortmarkers[markind]].get(node+remwidth,0.0)
               assert val == item  
        for tind,item in enumerate(quadvec):
            globmarkind,remwidth = tind / width, tind % width
            markind1, markind2 = 0, 0
            while True:
                globmarkind -= (len(sortmarkers) - 1- markind1)
                if globmarkind < 0:
                   break 
                markind1 += 1
            markind2 += (globmarkind + len(sortmarkers)) % len(sortmarkers)
            #print globmarkind, markind1, markind2
            if remwidth == 0:
               val = mark2pos2count[sortmarkers[markind1]].get(node,0.0) * mark2pos2count[sortmarkers[markind2]].get(node,0.0)
               assert val == item  
            else:
               val1 = sum(mark2pos2count[sortmarkers[markind1]].get(node+tval,0.0) for tval in [-1*remwidth, remwidth])
               val2 = sum(mark2pos2count[sortmarkers[markind2]].get(node+tval,0.0) for tval in [-1*remwidth, remwidth])
               assert val1 * val2 == item       
        return True

    @staticmethod
    def compareSol2dictVomm(solx,paramdict,sortmarkers,width,compcount):
        """compares sol vector to dictionary vomm case
        """
        for (keystr,start) in [("bound",0),("inside",len(solx)/2)]:
            for mind,marker in enumerate(sortmarkers):
                for twin in paramdict[keystr][marker].keys(): 
                    for tind,titem in enumerate(paramdict[keystr][marker][twin]):
                        assert abs(titem - solx[start+(mind*width*compcount)+((twin-1)*compcount)+tind]) < 0.001  
        return True
                       
    @staticmethod
    def testParamDict(paramdict,model,width,compcount):
        """tests param testParamDict
        Args:
           paramdict:
           model,width,compcount:
        Returns:
        """
        if model in ["linear","binary"]:
           for curpar in paramdict.keys():
               for markinfo in paramdict[curpar].keys():
                   assert len(paramdict[curpar][markinfo].keys()) == width 
                   for wind in paramdict[curpar][markinfo].keys():
                       assert type(paramdict[curpar][markinfo][wind]) != list
        elif model == "nonparam":
           for curpar in paramdict.keys():
               for markinfo in paramdict[curpar].keys():
                   assert len(paramdict[curpar][markinfo].keys()) == width 
                   for wind in paramdict[curpar][markinfo].keys():
                       assert len(paramdict[curpar][markinfo][wind]) == compcount                           
        return True

    @staticmethod
    def testDomainParamEstimateMemm(marklist,domlist,paramdict,sortmarkers,sidecoefs,nodecounts,params):
        """tests domain param estimate memm
        Args:
        Returns:
           bool: true or false
        """
        compcount = None
        if params['model'] == "linear" and params['infermodel'] == "double":
           [Xstart, ystart, Xend, yend] = sidecoefs
        elif params['model'] == "linear" and params['infermodel'] in ["single-memm","single-memm2"]:
           [Xterm, yterm] = sidecoefs
        elif params['model'] == "nonparam" and params['infermodel'] == "double":
           [stlogcoefs,stlincoefs,endlogcoefs,endlincoefs,stobjval,endobjval,stmuvec,endmuvec,compcount] = sidecoefs 
        elif params['model'] == "nonparam" and params['infermodel'] in ["single-memm","single-memm2"]:
           print "not done"
           exit(1)  
        assert TestSEDFMest.testPreParamDataVomm(marklist,domlist,sortmarkers,nodecounts,params)
        assert TestSEDFMest.testParamDict(paramdict,params['model'],params['width'],compcount)
        if params['model'] == "nonparam": 
           estobjval = TestSEDFMest.estNonParamObj(paramdict,marklist,domlist,sortmarkers,params,nodecount,compcount,stmuvec,endmuvec)
           assert abs(stobjval + endobjval - estobjval) <= 0.1
        elif params['model'] == "linear": 
           estobjval = TestSEDFMest.estParamEstObj(marklist,domlist,paramdict,sortmarkers,nodecounts,params)
           print estobjval
        return True

    @staticmethod
    def estParamObjVomm(marklist,domlist,paramdict,sortmarkers,nodecounts,params):
        """estimate objective for given paramdict for vomm
        Args:
           marklist,domlist,paramdict:
           sortmarkers,nodecounts,params:
        Returns:
           objval:
        """    
        method2run = "inner{0}".format({"linear":"param","binary":"param","nonparam":"nonparam"}[params["model"]]) 
        if params["model"] in ["binary","linear"]:         
           Xdom, ydom, maxcount = [], [], max(nodecounts)  
        elif params["model"] == "nonparam":
           Xdom, logcoefs  = [], [] 
        def processVec(startpos,endpos):  
            tcountvec,tcountvec2 = np.zeros((varcount/(2*compcount),),dtype=np.float), np.zeros((varcount/(2*compcount),),dtype=np.float)
            for bpos in [startpos,endpos]:
                tcountvec += np.array(HistoneUtilities.getCountVec(mark2pos2count,bpos,sortmarkers,params["width"],params['order']),dtype=np.float)
            for insnode in xrange(startpos+1,endpos):
                tcountvec2 += np.array(HistoneUtilities.getCountVec(mark2pos2count,insnode,sortmarkers,params["width"],params['order']),dtype=np.float)     
            return np.append(tcountvec, tcountvec2)
        def estParam():
            tval = np.dot(sentvec,stx) + np.dot(sentvec,endx)
            return math.log(math.exp(tval)+1.0) - tval
        def estNonparam():
            sentvec = processVec(start,end)
            countvec = HistoneUtilities.getCompPars(sentvec,compcount)
            logcoefs.append(countvec)                              
        for mainind,markinfo in enumerate(marklist):
            sorteddoms = sorted(domlist[mainind]) 
            allnodes = range(1,nodecounts[mainind]+1)
            inters = HistoneUtilities.getEmptyClusters(sorteddoms,allnodes)
            mark2pos2count = HistoneUtilities.processMarkData(markinfo,params["model"]) 
            for start,end in sorteddoms:
                locals()[method2run](1)
            for start,end in inters:
               locals()[method2run](0)
        if params["model"] in ["binary","linear"]:                             
           return Xdom,ydom
        elif params["model"] == "nonparam":  
           lincoefs =  -1.0 * reduce(lambda x1,y1: x1+y1,Xdom)
        #if TESTMODE:
        #   assert TestSEDFMest.testPreNonParamData(marklist,domlist,sortmarkers,varcount,nodecount,params,compcount,stlogcoefs,endlogcoefs)              
        #return lincoefs, logcoefs

        #def estDoubleBound():  
        #    tval1, tval2 = np.dot(countvec,stx), np.dot(countvec,endx)
        #    return (math.log(math.exp(tval1)+1.0) + math.log(math.exp(tval2)+1.0) - tval1 - tval2) 
        def estParam2Vec(keyw):
            xvec = [paramdict[keyw][mark][tw] for markind,mark in enumerate(sortmarkers) for tw in xrange(1,params['width']+1)]
            if params['order'] == 2:
               xvec += [paramdict[keyw][(mark1,mark2)][tw] for mark1,mark2 in itertools.combinations(sortmarkers,2) for tw in xrange(1,params['width']+1)]
            return xvec
        bound,inside = estParam2Vec("bound"), estParam2Vec("inside") 
        boundfunc = "est{0}Bound".format(params["infermodel"].replace("-","").capitalize())
        indomfunc = "est{0}Indom".format(params["infermodel"].replace("-","").capitalize()) 
        remfunc = "est{0}Rem".format(params["infermodel"].replace("-","").capitalize())         
        objval = 0.0 
        for mainind,markinfo in enumerate(marklist):
            sorteddoms = sorted(domlist[mainind])
            boundlocs,indomlocs,remlocs = HistoneUtilities.getLocs(sorteddoms,nodecounts[mainind])
            mark2pos2count = HistoneUtilities.processMarkData(markinfo)
            for node in boundlocs:
                countvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,params["width"],params['order']),dtype=np.float) 
                objval += locals()[boundfunc]()      
            for node in indomlocs:
                countvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,params["width"],params['order']),dtype=np.float)      
                objval += locals()[indomfunc]()   
            for node in remlocs:
                countvec = np.array(HistoneUtilities.getCountVec(mark2pos2count,node,sortmarkers,params["width"],params['order']),dtype=np.float)      
                objval += locals()[remfunc]()           
        objval += params['lambda'] * sum([abs(item) for item in termx])             
        return objval

    @staticmethod
    def testDomainParamEstimateVomm(marklist,domlist,paramdict,sortmarkers,sidecoefs,nodecounts,params,varcount):
        """tests domain param estimate vomm
        Args:
        Returns:
           bool: true or false
        """
        compcount = 1
        if params['model'] in ["linear","binary"] and params["order"] == 1:
          [Xdom,ydom,solx] = sidecoefs
        elif params['model'] in ["linear","binary"]:
          [Xdom,ydom,lincoefs,logcoefs,solx,termobjval,muvec] = sidecoefs
        elif params['model'] == "nonparam":
          [lincoefs,logcoefs,solx,objval,muvec,compcount] = sidecoefs
        assert TestSEDFMest.testPreParamDataVomm(marklist,domlist,sortmarkers,nodecounts,params)
        assert TestSEDFMest.testParamDict(paramdict,params['model'],params['width'],compcount)
        assert TestSEDFMest.compareSol2dictVomm(solx,paramdict,sortmarkers,params['width'],compcount)
        return True
        if params['model'] == "nonparam": 
           assert TestSEDFMest.testPreNonParamData(marklist,domlist,sortmarkers,varcount,nodecounts,params,compcount,logcoefs)   
           estobjval = TestSEDFMest.estNonParamObj(paramdict,marklist,domlist,sortmarkers,params,nodecount,compcount,stmuvec,endmuvec)
           assert abs(stobjval + endobjval - estobjval) <= 0.1
        elif params['model'] in ["binary","linear"] and params["order"] != 1:
           #def temploglikeEst(paramx,lincoefs,logcoefs,tmuvec):
           #    """negative log likelihood obj + penalty
           #    """
           #    tobjval = np.dot(paramx,lincoefs)
           #    templist = [np.dot(paramx,logcoef) for logcoef in logcoefs]
           #    tobjval += sum([tval if tval >= 10 else math.log(1.0 + math.exp(tval)) for tval in templist])
           #    tobjval += estPenaltyParam(paramx,tmuvec,singcount,width,curlam)
           #    return tobjval 
           #print "start"     
           #print temploglikeEst(res['x'],lincoefs,logcoefs,muvec)
           #print res['fun']
           #assert abs(temploglikeEst(res['x'],lincoefs,logcoefs,muvec) - res['fun']) < 0.1  
           for lcoef in lincoefs:
               assert lcoef >= 0
           estobjval = TestSEDFMest.estParamEstObj(marklist,domlist,paramdict,sortmarkers,nodecounts,params)
           print estobjval
           exit(1)
        return True

  

