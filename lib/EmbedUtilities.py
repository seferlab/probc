#Methods used
import os
import sys
import time
import numpy as np
import networkx as nx
import gzip
import itertools
import math
import random


def loadSampleFileAll(outpath):
    """loads sample file
    Args:
       outpath:
    Returns:
       vals:
    """
    with gzip.open(outpath,"r") as infile:
        vals = [float(line.rstrip()) for line in infile]
    return vals


def saveSampleFileAll(vals,statpath):
    """saves sampl
    Args:
       comp2vals:
       outpath:
    """
    with gzip.open(statpath,"w") as outfile:
         outfile.write("\n".join([str(val) for val in vals])+"\n")


def saveSampleFile(comp2vals,statpath):
    """saves sample file
    Args:
       comp2vals:
       outpath:
    """
    with gzip.open(statpath,"w") as outfile:
        for comp in sorted(comp2vals.keys()):
            outfile.write("comp\t{0}\n".format(comp))
            outfile.write("\n".join([str(val) for val in comp2vals[comp]])+"\n")

                     
def loadSampleFile(outpath):
    """loads sample file
    Args:
       outpath:
    Returns:
       comp2vals:
    """
    comp2vals = {}
    curcomp = None
    with gzip.open(outpath,"r") as infile:
        for line in infile:
            line = line.rstrip()
            if line.startswith("comp"):
               curcomp = int(line.replace("comp",""))
               comp2vals[curcomp] = []
               continue
            comp2vals[curcomp].append(float(line))
    return comp2vals


def getCumulative(scale2doms):
    """returns cumulative scale to domains
    Args:
       scale2doms:
    Returns:
       cumscale2doms:
    """
    cumscale2doms = {}
    for scale in scale2doms.keys():
        domset = set()
        for scale2 in scale2doms.keys():
            if scale2 <= scale:
               domset |= set(scale2doms[scale2])
        cumscale2doms[scale] = set(domset)
    return cumscale2doms 


def genKernelCoefMat(freqmat,algopar):
    """generates kernel coefs
    Args:
    Returns:
    """
    assert algopar["kernel"] in ["exp","gauss"]
    if algopar["kernel"] == "exp":
       distfunc = lambda ind1,ind2: math.exp(algopar["kerncoef"]*(abs(ind1-ind2)+1))  
    elif algopar["kernel"] == "gauss":
       distfunc = lambda ind1,ind2: math.exp((0.5/algopar["kerncoef"])*((ind1-ind2)**2))  
    coefmat = np.zeros(np.shape(freqmat),dtype=np.float)
    for ind1 in xrange(np.shape(coefmat)[0]):
        for ind2 in xrange(np.shape(coefmat)[1]):
            try:
               dif = distfunc(ind1,ind2)  
               if dif <= 10000000000:
                  coefmat[ind1,ind2] = 1.0/dif
            except:
               pass
    return coefmat


def estDistances(in2loc):
    """estimates distances from index to location mapping
    Args:
       in2loc: index to locations
    Returns:
       distmat: 
    """
    if type(in2loc) == dict:
       matlen = len(in2loc.keys())
    elif type(in2loc) == list:
       matlen = len(in2loc)    
    distmat = np.zeros((matlen,matlen),dtype=np.float)
    for ind1 in xrange(matlen):
        if type(in2loc) == dict:
           x1,y1,z1 = in2loc[ind1+1]
        elif type(in2loc) == list:
           x1,y1,z1 = in2loc[ind1] 
        for ind2 in xrange(ind1+1,matlen):
            if type(in2loc) == dict:
               x2,y2,z2 = in2loc[ind2+1]
            elif type(in2loc) == list:
               x2,y2,z2 = in2loc[ind2] 
            dist = math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
            distmat[ind1,ind2] = dist
            distmat[ind2,ind1] = dist
    return distmat

    
def preprocDomains(alldomains,domains):
    """preprocess domains
    Args:
       alldomains:
       domains:
    Returns:
       sentdoms:
    """
    sentdoms = set(domains)
    for start,end in alldomains:
        flag = False
        for start1,end1 in domains:
            if abs(start1-start) + abs(end1-end) <= 2:
               flag = True
               break   
        if flag:         
           sentdoms.add((start,end))
    return list(sentdoms)
   

def getPosDomains(freqmat,domains):
    """get all possible intervals
    Args:
       freqmat:
       domains:
    Returns:
       posdomains:
    """
    posdomains = [(node,node+intlen) for node in xrange(np.shape(freqmat)[0]) for intlen in xrange(2,np.shape(freqmat)[0]-node)]
    return posdomains


def getAlgoStr(method,objtype,algo,scalemode,algopar):
    """gets algo str
    Args:
       method,objtype,algo,scalemode,algopar:
    Returns:
       algostr:
    """
    return "{0}-{1}-{2}-{3}-{4}".format(method,objtype,algo,scalemode,"_".join(str(val) for item in sorted(algopar.keys()) for val in [item,algopar[item]]))


def getNoiseStr(noisetype,noise):
    """gets noise string
    Args:
       noisetype:
       noise:
    Returns:
       noisestr:
    """     
    noisestr = "{0}_{1}".format(noisetype,noise)
    if noise == 0.0:
       noisestr = "{0}".format(noise) 
    return noisestr


def addPriorFracObj(xdict,domains,predomains,lamcoef):
    """add prior for frac variables
    Args:
       xdict:
       domains,predomains:
       lamcoef:
    Returns:
       addfracobj:
    """
    dom2index = {domains[index]: index for index in xrange(len(domains))}
    #addfracobj = -1.0*lamcoef*sum([xdict[(dom2index[predom],comp,scale)] for predom in predomains for comp in xrange(compcount) for scale in scales if xdict.has_key((dom2index[predom],comp,scale))])
    predomins = [dom2index[predomain] for predomain in predomains]
    addfracobj = -1.0*lamcoef*sum([xdict[(domin,comp,scale)] for domin,comp,scale in xdict.keys() if domin in predomins])
    return addfracobj


def addPriorObj(comp2dominds,domains,predomains,lamcoef):
    """add prior to obj for comp2domain mapping
    Args:
       comp2dominds:
       domains,predomains:
       lamcoef:
    Returns:
       priorobj:
    """
    preset = set(predomains)
    return -1.0*lamcoef*sum([len(set(domains[domin] for domin in comp2dominds[comp]).intersection(preset)) for comp in comp2dominds.keys()])


def addPriorCoef(b,lamcoef,predomains,compcount,domains,var2index,minmax):
    """adds prior coef
    Args:
       b,lamcoef:
       predomains: file domains
       compcount:
       domains: all domains
       var2index:
       minmax: min or max
    Returns:
    """
    coef = -1.0
    if minmax == "max":
       coef = 1.0
    dom2index = {domains[index]: index for index in xrange(len(domains))}
    usescales = set(scale for domin,comp,scale in var2index.keys())
    addcount = 0
    for start,end in predomains:
        for comp in xrange(compcount):
            for scale in usescales:
                varstr = (dom2index[(start,end)],comp,scale)
                if var2index.has_key(varstr):
                   b[var2index[varstr]] += lamcoef*coef*0.5 #because of 2 multiplier!!


def getRatio(freqmat,comp2dominds,comp2scale,domains,kernmat=None):
    """returns solution ratio 
    Args:
       freqmat:
       comp2dominds,comp2scale
       domains:
       kernmat: 
    Returns:
       ratio:
       newobjval:
    """
    secondmat = np.zeros(np.shape(freqmat),dtype=np.float)
    for comp in comp2dominds.keys():
        for domin in comp2dominds[comp]:
            start,end = domains[domin]
            secondmat[start:end+1,start:end+1] += comp2scale[comp]      
    difmat = freqmat - secondmat
    if kernmat == None:
       newobj = sum([difmat[ind1,ind2]**2 for ind1 in xrange(np.shape(difmat)[0]) for ind2 in xrange(np.shape(difmat)[1])])
       freqsum = sum([freqmat[ind1,ind2]**2 for ind1 in xrange(np.shape(freqmat)[0]) for ind2 in xrange(np.shape(freqmat)[1])])
    else:
       newobj = sum([(difmat[ind1,ind2]**2)*kernmat[ind1,ind2] for ind1 in xrange(np.shape(difmat)[0]) for ind2 in xrange(np.shape(difmat)[1])])     
       freqsum = sum([(freqmat[ind1,ind2]**2)*kernmat[ind1,ind2] for ind1 in xrange(np.shape(freqmat)[0]) for ind2 in xrange(np.shape(freqmat)[1])])
    return newobj/freqsum, newobj


def makeDeconOutput(outprefix,domains,comp2dominds,comp2scale,metadata):
    """make decon outputs
    Args:
       outprefix:
       domains:
       comp2dominds:
       comp2scale:
       metadata 
    Returns:
    """
    splitted = outprefix.split("/")
    if len(splitted) == 2:
       outfolder = "/".join(splitted[0:-1])
       if not os.path.exists(outfolder):
          os.makedirs(outfolder)
    deconoutfile = "{0}_decon.txt".format(outprefix)
    objoutfile = "{0}_objscore.txt".format(outprefix)
    print "Writing output"
    tdomains = [(start+1,end+1) for start, end in domains]
    writeDeconOut(comp2dominds,comp2scale,tdomains,deconoutfile)
    writeDeconMeta(metadata,objoutfile)


def writeCompcountFile(compcountpath,compcount):
    """write class count file
    Args:
       compcountpath:
       compcount:
    Returns:
    """        
    with open(compcountpath,"w") as outfile:
         outfile.write("{0}\n".format(compcount))

def readCompcountFile(compcountpath):
    """reads class count file
    Args:
       compcountpath:
    Returns:
       compcount:
    """        
    with open(compcountpath,"r") as infile:
       for line in infile:
           compcount = int(line.rstrip())
    return compcount     

def findDomainCliqueDecomp(domains,interdom):
    """finds maximal clique decomposition 
    Args:
       domains:
       interdom:
    Returns:
       cliques:
    """
    G = nx.Graph()
    for dom in domains:
        G.add_node(dom)
    for dom1,dom2 in interdom:
        G.add_edge(dom1,dom2)
    return findCliqueDecomp(G)

    
def findCliqueDecomp(interG):
    """finds maximal clique decomposition 
    Args:
       interG
    Returns:
       cliques:
    """
    tinterG = nx.Graph(interG)
    return list(map(list,nx.find_cliques(tinterG)))



#xdict,freqmat,node2dom,scales,compcount):
def estMatrices(freqmat,scales,compcount,domains,interdom):
    """generates sdp matrix coefs
    Args:
       freqmat: frequency matrix
       scales: set of scales
       compcount: number of components
       domains: all domains
       interdom: intersecting domains
    Returns:
       objstr: 
    """
    #A = np.zeros((),dtype=np.float)
    dom2index = {domains[index]:index for index in xrange(len(domains))}
    var2index, index2var, varcount = {}, [], 0
    for dom,comp,scale in list(itertools.product(domains,range(compcount),scales)):
        var2index[(dom2index[dom],comp,scale)] = varcount
        varcount += 1
        index2var.append((dom2index[dom],comp,scale))
    coefmat = np.zeros((varcount,varcount),dtype=np.float) #coefmat = scipy.sparse.bsr_matrix((varcount,varcount), dtype=np.int)
    bvec = [0] * varcount
    pairs = [(comp,scale) for comp in xrange(compcount) for scale in scales]
    for dom1,dom2 in interdom:
        domin1 = dom2index[dom1]
        domin2 = dom2index[dom2]
        interlen = min(dom1[1],dom2[1])-max(dom1[0],dom2[0]) + 1
        for (comp1,scale1),(comp2,scale2) in itertools.product(pairs,pairs):
            coefmat[var2index[(domin1,comp1,scale1)],var2index[(domin2,comp2,scale2)]] += (interlen**2)*scale1*scale2
            coefmat[var2index[(domin2,comp2,scale2)],var2index[(domin1,comp1,scale1)]] += (interlen**2)*scale1*scale2
    for dom in domains:
        domin = dom2index[dom]
        qcoef = (dom[1]-dom[0]+1)**2
        fsum = np.sum(freqmat[dom[0]:dom[1]+1,dom[0]:dom[1]+1])
        for (comp1,scale1),(comp2,scale2) in itertools.product(pairs,pairs):
            coefmat[var2index[(domin,comp1,scale1)],var2index[(domin,comp2,scale2)]] += 0.5*qcoef*scale1*scale2
            coefmat[var2index[(domin,comp2,scale2)],var2index[(domin,comp1,scale1)]] += 0.5*qcoef*scale1*scale2
        for comp,scale in pairs:
            #coefmat[var2index[(domin,comp,scale)],var2index[(domin,comp,scale)]] += -2.0*scale*fsum 
            bvec[var2index[(domin,comp,scale)]] += -2.0*scale*fsum
    return coefmat, index2var


def mapVarsSpec(domains,compcount,comp2scale):
    """maps vars to matrix indices specific
    Args:
       domains:
       compcount:
       comp2scale:
    Returns:
       var2index,index2var:
    """
    var2index,index2var,index = {},[],0
    for domin in xrange(len(domains)):
        for comp in xrange(compcount):
            var2index[(domin,comp,comp2scale[comp])] = index
            index += 1
            index2var.append((domin,comp,comp2scale[comp]))
    return var2index,index2var


def mapVars2(comp2dominds,compcount,scales):
    """maps vars to matrix indices
    Args:
       comp2dominds:
       compcount:
       scales:
    Returns:
       var2index,index2var:
    """
    var2index,index2var,index = {},[],0
    for comp in comp2dominds.keys():
        for domin in comp2dominds[comp]:
            for scale in scales:
                var2index[(domin,comp,scale)] = index
                index += 1
                index2var.append((domin,comp,scale))
    return var2index,index2var


def mapVars(domains,compcount,scales):
    """maps vars to matrix indices
    """
    var2index,index2var,index = {},[],0
    for domin in xrange(len(domains)):
        for comp in xrange(compcount):
            for scale in scales:
                var2index[(domin,comp,scale)] = index
                index += 1
                index2var.append((domin,comp,scale))
    return var2index,index2var


def genCoefsSecond(freqmat,comp2dominds,domains,interdom,scales,var2index):
    """generates coefs for second part
    Args:
       freqmat: frequency matrix
       comp2dominds: 
       domains: all domains
       interdom: intersecting domains
       scales:
       var2index:
    Returns:
       objstr: 
    """
    dom2index = {domains[index]:index for index in xrange(len(domains))}
    varcount = len(var2index.keys())
    A = np.zeros((varcount,varcount),dtype=np.float)
    b = np.array([0.0] * varcount)
    pairs = list(itertools.product(range(len(comp2dominds.keys())),list(scales)))
    for dom1,dom2 in interdom:
        domin1 = dom2index[dom1]
        domin2 = dom2index[dom2]
        qcoef = (min(dom1[1],dom2[1])-max(dom1[0],dom2[0]) + 1)**2
        for comp1,scale1 in pairs:
            for comp2,scale2 in pairs:
                if domin1 in comp2dominds[comp1] and domin2 in comp2dominds[comp2]:
                   A[var2index[(0,comp1,scale1)],var2index[(0,comp2,scale2)]] += qcoef*scale1*scale2
                   A[var2index[(0,comp2,scale2)],var2index[(0,comp1,scale1)]] += qcoef*scale1*scale2
    for dom in domains:
        domin = dom2index[dom]
        qcoef = (dom[1]-dom[0]+1)**2
        for comp1,scale1 in pairs:
            for comp2,scale2 in pairs:
                if domin in comp2dominds[comp1] and domin in comp2dominds[comp2]:
                   A[var2index[(0,comp1,scale1)],var2index[(0,comp2,scale2)]] += 0.5*qcoef*scale1*scale2
                   A[var2index[(0,comp2,scale2)],var2index[(0,comp1,scale1)]] += 0.5*qcoef*scale1*scale2
        fsum = np.sum(freqmat[dom[0]:dom[1]+1,dom[0]:dom[1]+1])
        for comp,scale in pairs:
            if domin in comp2dominds[comp]:
               b[var2index[(0,comp,scale)]] -= fsum*scale
    return A,b


def genCoefs4(freqmat,compcount,domains,interdom,scales,var2index):
    """generates coefs
    Args:
       freqmat: frequency matrix
       compcount: number of components
       domains: all domains
       interdom: intersecting domains
       scales:
       var2index:
    Returns:
       objstr: 
    """
    dom2index = {domains[index]:index for index in xrange(len(domains))}
    varcount = len(domains) * compcount * len(scales)
    A = np.zeros((varcount,varcount),dtype=np.float)
    b = np.array([0.0] * varcount)
    pairs = list(itertools.product(range(compcount),list(scales)))
    for dom1,dom2 in interdom:
        domin1 = dom2index[dom1]
        domin2 = dom2index[dom2]
        qcoef = (min(dom1[1],dom2[1])-max(dom1[0],dom2[0]) + 1)**2
        for comp1,scale1 in pairs:
            for comp2,scale2 in pairs:
                A[var2index[(domin1,comp1,scale1)],var2index[(domin2,comp2,scale2)]] += qcoef
                A[var2index[(domin2,comp2,scale2)],var2index[(domin1,comp1,scale1)]] += qcoef
    for dom in domains:
        domin = dom2index[dom]
        qcoef = (dom[1]-dom[0]+1)**2
        for comp1,scale1 in pairs:
            for comp2,scale2 in pairs:
                A[var2index[(domin,comp1,scale1)],var2index[(domin,comp2,scale2)]] += 0.5*qcoef
                A[var2index[(domin,comp2,scale2)],var2index[(domin,comp1,scale1)]] += 0.5*qcoef
        fsum = np.sum(freqmat[dom[0]:dom[1]+1,dom[0]:dom[1]+1])
        for comp,scale in pairs: 
            b[var2index[(domin,comp,scale)]] -= fsum
    return A,b




def genCoefs3(freqmat,compcount,domains,interdom,scales,var2index,comp2scales):
    """generates coefs 3
    Args:
       freqmat: frequency matrix
       compcount: number of components
       domains: all domains
       interdom: intersecting domains
       scales:
       var2index:
       comp2scales:
    Returns:
       objstr: 
    """
    dom2index = {domains[index]:index for index in xrange(len(domains))}
    #scalesum = sum(scales)
    varcount = len(domains) * compcount * len(scales)
    A = np.zeros((varcount,varcount),dtype=np.float)
    b = np.array([0.0] * varcount)
    pairs = list(itertools.product(range(compcount),list(scales)))
    for dom1,dom2 in interdom:
        domin1 = dom2index[dom1]
        domin2 = dom2index[dom2]
        qcoef = (min(dom1[1],dom2[1])-max(dom1[0],dom2[0]) + 1)**2
        for comp1,comp2 in itertools.product(range(compcount),range(compcount)):
            scale1,scale2 = comp2scale[comp1], comp2scale[comp2]
            A[var2index[(domin1,comp1,scale1)],var2index[(domin2,comp2,scale2)]] += qcoef
            A[var2index[(domin2,comp2,scale2)],var2index[(domin1,comp1,scale1)]] += qcoef
    for dom in domains:
        domin = dom2index[dom]
        qcoef = (dom[1]-dom[0]+1)**2
        for comp1,comp2 in itertools.product(range(compcount),range(compcount)):    
            scale1,scale2 = comp2scale[comp1], comp2scale[comp2]
            A[var2index[(domin,comp1,scale1)],var2index[(domin,comp2,scale2)]] += 0.5*qcoef
            A[var2index[(domin,comp2,scale2)],var2index[(domin,comp1,scale1)]] += 0.5*qcoef
        fsum = np.sum(freqmat[dom[0]:dom[1]+1,dom[0]:dom[1]+1])
        for comp,scale in pairs: 
            b[var2index[(domin,comp,scale)]] -= fsum
    return A,b


def genCoefs(freqmat,compcount,domains,interdom,var2index,kernmat=None):
    """generates coefs
    Args:
       freqmat: frequency matrix
       compcount: number of components
       domains: all domains
       interdom: intersecting domains
       var2index:
       kernmat:
    Returns:
       objstr: 
    """
    if kernmat != None:
       coefest = lambda dom1,dom2: np.sum(kernmat[max(dom1[0],dom2[0]):min(dom1[1],dom2[1])+1,max(dom1[0],dom2[0]):min(dom1[1],dom2[1])+1])
    elif kernmat == None:
       coefest = lambda dom1,dom2: (min(dom1[1],dom2[1])-max(dom1[0],dom2[0]) + 1)**2
    dom2pairs = {domin:set() for domin in xrange(len(domains))}
    for domin,comp,scale in var2index.keys():
        dom2pairs[domin].add((comp,scale))
    dom2index = {domains[index]:index for index in xrange(len(domains))}
    varcount = len(var2index.keys())
    A = np.zeros((varcount,varcount),dtype=np.float)
    b = np.array([0.0] * varcount)
    for dom1,dom2 in interdom:
        domin1 = dom2index[dom1]
        domin2 = dom2index[dom2]
        qcoef = coefest(dom1,dom2)
        for (comp1,scale1),(comp2,scale2) in itertools.product(dom2pairs[domin1],dom2pairs[domin2]):
            A[var2index[(domin1,comp1,scale1)],var2index[(domin2,comp2,scale2)]] += qcoef*scale1*scale2
            A[var2index[(domin2,comp2,scale2)],var2index[(domin1,comp1,scale1)]] += qcoef*scale1*scale2
    for dom in domains:
        domin = dom2index[dom]
        qcoef = coefest(dom,dom)
        for (comp1,scale1),(comp2,scale2) in itertools.product(dom2pairs[domin],dom2pairs[domin]):
            A[var2index[(domin,comp1,scale1)],var2index[(domin,comp2,scale2)]] += 0.5*qcoef*scale1*scale2
            A[var2index[(domin,comp2,scale2)],var2index[(domin,comp1,scale1)]] += 0.5*qcoef*scale1*scale2
        if kernmat == None:
           fsum = np.sum(freqmat[dom[0]:dom[1]+1,dom[0]:dom[1]+1])
        else:
           fsum = np.sum(np.multiply(kernmat[dom[0]:dom[1]+1,dom[0]:dom[1]+1],freqmat[dom[0]:dom[1]+1,dom[0]:dom[1]+1]))     
        for comp,scale in dom2pairs[domin]:
            b[var2index[(domin,comp,scale)]] -= fsum*scale
    return A,b


def estFracObjective(xdict,freqmat,node2dom,scales,compcount,kernmat=None):
    """estimates objective function when xdict may be fractional
    Args:
       xdict:
       freqmat:
       node2dom:
       scales:
       compcount:
       kernmat:
    Returns:
       objval:
    """
    objval = 0.0
    for in1 in xrange(np.shape(freqmat)[0]):
        for in2 in xrange(np.shape(freqmat)[1]):
            cursum = freqmat[in1,in2]
            domset = node2dom[in1].intersection(node2dom[in2])
            cursum -= sum([scale*xdict[(dom,comp,scale)] for dom in domset for comp in xrange(compcount) for scale in scales if xdict.has_key((dom,comp,scale))])
            if kernmat == None:
               objval += cursum**2  
            else:      
               objval += (cursum**2)*kernmat[in1,in2]                      
    return objval


def estFracObjectiveMat(A,b,fqsum,var2index,xdict):
    """estimates objective function in matrix form
    Args:
       A:
       b:
       fqsum:
       var2index:
       xdict:
       scales:
    Returns:
       matobjval:
    """
    xvec = np.array([0.0] * np.shape(b)[0])
    for domin,comp,scale in xdict.keys():
        xvec[var2index[(domin,comp,scale)]] = xdict[(domin,comp,scale)]
    matobjval = np.dot(xvec.transpose(),np.dot(A,xvec)) + 2*np.dot(b,xvec) + fqsum
    return matobjval


def getnode2dom(freqmat,domains):
    """return node 2 domain mapping
    Args:
       freqmat:
       domains:
    Returns:
       node2dom:
    """
    node2dom = {node:set() for node in xrange(np.shape(freqmat)[0])}
    for ind,dom in list(enumerate(domains)):
        for node in xrange(dom[0],dom[1]+1):
            node2dom[node].add(ind)
    return node2dom


def inter2Graph(interdom,domains):
    """returns inter domain graph
    Args:
       interdom: list of domains
       domains:
    Returns:
       G: 
    """
    G = nx.Graph()
    for dom in domains:
        G.add_node(dom)
    for dom1,dom2 in interdom:
        G.add_edge(dom1,dom2)
    return G
 
def getInterDomain(domains):
    """returns intersecting domain set
    Args:
       domains: list of domains
    Returns:
       interset: 
    """
    intersect = lambda (s1,e1), (s2,e2): False if (e1 < s2 or e2 < s1) else True
    enumdomains = list(enumerate(domains))
    return set((dom1,enumdomains[in2][1]) for in1,dom1 in enumdomains for in2 in xrange(in1+1,len(enumdomains)) if intersect(dom1,enumdomains[in2][1]))


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


def readDomainFile(domfile):
    """reads domain file
    Args:
       domfile:
    Returns:
       domains:
    """
    domains = []
    with open(domfile,"r") as infile:
        for line in infile:
            splitted = line.rstrip().split(",")
            domains.append((int(splitted[0]),int(splitted[1])))
    return domains


def readFreqMatrixSize(freqpath):
    """reads freq matrix size without reading data
    Args:
       freqpath:
    Returns:
       nodecount:
    """
    if freqpath.endswith(".gz"):
       compress = True
    nodecount = 0
    if compress:
       with gzip.open(freqpath,"r") as infile:
          for line in infile:
              nodecount += 1
    else:
       with open(freqpath,"r") as infile:
          for line in infile:
              nodecount += 1       
    return nodecount   


def readFreqFile(freqfile):
    """reads input matrix file in gz format
    Args:
       freqfile:
    Returns:
       freqmat:
       nodenames:
       in2pos: index to genome start end locations~(list)
    """    
    nodenames,in2pos = [], []
    compress = False
    if freqfile.endswith(".gz"):
       compress = True
    if compress:
       with gzip.open(freqfile,"r") as infile:
            for line in infile:
                node,start,end = line.rstrip().split("\t")[0:3]
                nodenames.append(node)
                in2pos.append((int(start),int(end)))    
    else:
       with open(freqfile,"r") as infile:
            for line in infile:
                node,start,end = line.rstrip().split("\t")[0:3]
                nodenames.append(node)
                in2pos.append((int(start),int(end)))             
    freqmat = np.zeros((len(nodenames),len(nodenames)),dtype=np.float)
    index = 0 
    if compress:      
       with gzip.open(freqfile,"r") as infile:
            for line in infile:
                parts = line.rstrip().split("\t")
                for index2 in xrange(len(nodenames)):
                    freqmat[index,index2] = float(parts[3+index2])
                index+=1
    else:
       with open(freqfile,"r") as infile:
            for line in infile:
                parts = line.rstrip().split("\t")
                for index2 in xrange(len(nodenames)):
                    freqmat[index,index2] = float(parts[3+index2])
                index+=1         
    return freqmat,nodenames,in2pos


def writeFreqData(freqmat,freqfile,in2loc=None):
    """writes frequency data to file
    Args:
       freqmat:
       freqfile:
       in2loc:
    Returns:
    """
    if in2loc == None:
       in2loc = {ind:[ind,ind+1] for ind in xrange(np.shape(freqmat)[0])}
    with gzip.open(freqfile,"w") as outfile:
        for in1 in xrange(np.shape(freqmat)[0]):
            linestr = "node{0}\t{1}\t{2}\t".format(in1,in2loc[in1][0],in2loc[in1][1]) + "\t".join(["{0}".format(freqmat[in1,in2]) for in2 in xrange(np.shape(freqmat)[1])])
            outfile.write(linestr+"\n")            

            
def writeDeconMeta(metadata,objoutfile):
    """writes deconvolution meta file
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

            
def readDeconMeta(deconfile):
    """read deconvolution meta file
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
                        
        
def readDeconOut(deconoutfile):
    """reads deconvolution output
    Args:
       deconoutfile:
    Returns:
       comp2domains:
       comp2scale:
    """
    comp2domains,comp2scale = {}, {}
    with open(deconoutfile,"r") as infile:
        index = 0
        for line in infile:
            parts = line.rstrip().split("\t")
            comp2scale[index] = float(parts[0])
            comp2domains[index] = set()
            for part in parts[1:]:
                start,end = [int(item) for item in part.split(",")]
                comp2domains[index].add((start,end))
            index += 1    
    return comp2domains,comp2scale

        
def writeDeconOut(comp2dominds,comp2scale,domains,deconoutfile):
    """writes deconvolution output
    Args:
       comp2dominds: list of tuples
       comp2scale:
       domains:
       outensfile: output ensemble file
    Returns:
    """
    with open(deconoutfile,"w") as outfile:
        for comp in comp2dominds.keys():
            linestr = "\t".join([str(comp2scale[comp])] + ["{0},{1}".format(domains[domin][0],domains[domin][1]) for domin in comp2dominds[comp]])
            outfile.write(linestr+"\n") 
     
            
def readCplexOut(outfile,specific=[]):
    """reads CPLEX output file and returns only SPECIFIC variable values as dictionary
    Args:
       outfile: CPLEX output file
       specific: specific variable prefixes such as x
    Returns:
       retvalues: variable-value dictionary
    """
    retvalues = {}
    varflag = False
    objval = None
    with open(outfile,"r") as file:
        for line in file:
            line = line.rstrip()
            if not varflag and line.find("Objective")!=-1 and (line.find("- Non-optimal:")!=-1 or line.find("- Optimal:")!=-1 or line.find(" - Integer")!=-1):
               objval = float(line.split("=")[1])
               continue
            if not varflag and line.find("Variable Name")!=-1 and line.find("Solution Value")!=-1:
               varflag=True
               continue
            if varflag:
               for varname in specific: 
                   if line.startswith(varname):
                      key,value = line.split()
                      retvalues[key] = float(value)
                      break
    return retvalues,objval 


def runCplexCode(consstr,objstr,boundstr,varstr,runfolder,outmethod):
    """Runs cplex code
    Args:
        consstr: constraint string
        objstr: objective function string
        boundstr: boundary string
        varstr: variable string
        runfolder: run folder
        outmethod = function to be run before returning output
    Returns:
        xdict:
        ydict:
    """
    if not os.path.exists(runfolder):
       os.makedirs(runfolder)
    PNUM = 1 #processor count
    filepref = "cplexrun"
    outlppath = "{0}/{1}.lp".format(runfolder,filepref)
    with open(outlppath,"w") as file:
       file.write(objstr+"\n")
       file.write(consstr)
       file.write(boundstr+"\n")
       file.write(varstr+"\n")
       file.write("End\n")
    cplexoutpath = "{0}/{1}.lpout".format(runfolder,filepref)
    cplexscriptpath = "{0}/{1}.script".format(runfolder,filepref)
    with open(cplexscriptpath,"w") as file:
       file.write("read {0}\n".format(outlppath))
       file.write("set threads {0}\n".format(PNUM))
       file.write("optimize\n")
       file.write("display solution objective\n")
       file.write("display solution variables -\n")  
    t1 = time.time()
    code="cplex < {0} > {1}".format(cplexscriptpath,cplexoutpath)
    os.system(code)
    t2 = time.time()
    print "Cplex solved problem in {0} seconds".format(t2-t1)
    returns = outmethod(cplexoutpath)
    os.system("rm -rf {0}".format(runfolder))
    return returns                                 
