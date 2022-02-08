#Real Data Prepare
import networkx as nx
import numpy as np
import os
import sys
import math
import random
import EmbedUtilities
import itertools
import myutilities as myutil


def readMatrixFormat(filepath):
    """reads pure matrix format
    """
    linecount = 0
    with open(filepath,"r") as infile:
       for line in infile:
           line = line.rstrip()
           linecount += 1
    freqmat = np.zeros((linecount,linecount),dtype=np.float)
    rowin = 0
    with open(filepath,"r") as infile:
       for line in infile:
           parts = line.rstrip().split("\t")
           for colind in xrange(len(parts)):
               freqmat[rowin,colind] = float(parts[colind])
           rowin += 1
    return freqmat


def readArmatusOut(armaoutpath):
    """reads armatus out file
    Args:
       armaoutpath:
    Returns:
       domains:
    """
    domains = set()
    with open(armaoutpath,"r") as infile:
        for line in infile:
            start,end = [int(item) for item in line.rstrip().split("\t")[1:]]
            domains.add((start,end))            
    return domains


def runArmatus(filepath,maxscale,outprefix,armaoutfolder,ARMATUSPATH,stepsize=0.05,topcount=1):
    """Runs Armatus
    Args:
       filepath:
       maxscale:
       outprefix:
       armaoutfolder:
       ARMATUSPATH:
       topcount: suboptimal solution count
    Returns:
       alldomains:
    """
    intersect = lambda (s1,e1), (s2,e2): False if (e1 < s2 or e2 < s1) else True
    if not os.path.exists(armaoutfolder):
       os.makedirs(armaoutfolder)
    code = "{0}/armatus -i {1} -g {2} -o {3}/{4} -k {5} -s {6} -m".format(ARMATUSPATH,filepath,maxscale,armaoutfolder,outprefix,topcount,stepsize)
    os.system(code)
    scale2doms = {}
    outpaths = ["{0}/{1}".format(armaoutfolder,tarmaoutfile) for tarmaoutfile in myutil.listfiles(armaoutfolder) if tarmaoutfile.find("gamma")!=-1]
    for outpath in outpaths:
        domains = readArmatusOut(outpath)
        scale = float(".".join(outpath.split(outprefix+".gamma.")[-1].split(".")[0:2]))
        scale2doms.setdefault(scale,set())
        scale2doms[scale] |= set(domains)
        for dom1,dom2 in itertools.combinations(list(domains),2):
            assert not intersect(dom1,dom2)
    os.system("rm -rf {0}".format(armaoutfolder))
    for scale in scale2doms.keys():
        for dom1,dom2 in itertools.combinations(list(scale2doms[scale]),2):
            assert dom1 != dom2
    return scale2doms


def runArmatusConsensus(filepath,maxscale,outprefix,armaoutfolder,ARMATUSPATH,topcount=1,stepsize=0.05):
    """Runs Armatus only for consensus
    Args:
       filepath:
       maxscale:
       outprefix:
       armaoutfolder:
       ARMATUSPATH:
       topcount:
       stepsize:
    Returns:
       consdomains: consensus domains
    """
    intersect = lambda (s1,e1), (s2,e2): False if (e1 < s2 or e2 < s1) else True
    if not os.path.exists(armaoutfolder):
       os.makedirs(armaoutfolder)
    code = "{0}/armatus -i {1} -g {2} -o {3}/{4} -m -k {5} -s {6}".format(ARMATUSPATH,filepath,maxscale,armaoutfolder,outprefix,topcount,stepsize)
    os.system(code)
    outpath = "{0}/{1}.consensus.txt".format(armaoutfolder,outprefix)
    consdomains = readArmatusOut(outpath)
    for dom1,dom2 in itertools.combinations(list(consdomains),2):
        assert not intersect(dom1,dom2)
    os.system("rm -rf {0}".format(armaoutfolder))            
    return consdomains


def runArmatusAsMethod(filepath,maxscale,outprefix,armaoutfolder,ARMATUSPATH,compcount):
    """Runs Armatus as method
    Args:
       filepath:
       maxscale:
       outprefix:
       armaoutfolder:
       ARMATUSPATH:
       compcount: suboptimal solution count
    Returns:
       alldomains:
    """
    intersect = lambda (s1,e1), (s2,e2): False if (e1 < s2 or e2 < s1) else True
    if not os.path.exists(armaoutfolder):
       os.makedirs(armaoutfolder)
    code = "{0}/armatus -i {1} -g {2} -o {3}/{4} -k {5} -m".format(ARMATUSPATH,filepath,maxscale,armaoutfolder,outprefix,compcount)
    os.system(code)
    outpaths = ["{0}/{1}".format(armaoutfolder,tarmaoutfile) for tarmaoutfile in myutil.listfiles(armaoutfolder) if tarmaoutfile.find("gamma")!=-1]
    scale2comp2doms = {}
    for outpath in outpaths:
        domains = readArmatusOut(outpath)
        scale = float(".".join(outpath.split(outprefix+".gamma.")[-1].split(".")[0:2]))
        comp = int(outpath.split(outprefix+".gamma.")[-1].split(".")[2])
        scale2comp2doms.setdefault(scale,{})
        scale2comp2doms[scale][comp] = set(domains)
    os.system("rm -rf {0}".format(armaoutfolder))
    #for scale in sorted(scale2comp2doms.keys()):
    #    print scale
    #    for comp in scale2comp2doms[scale].keys():
    #        print comp,scale2comp2doms[scale][comp]
    #        for dom1,dom2 in itertools.combinations(list(scale2comp2doms[scale][comp]),2):
    #            assert dom1 != dom2
    #print scale2comp2doms.keys()            
    return scale2comp2doms

