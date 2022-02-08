import os
import networkx as nx
import sys
import string
import random
import csv
#import Pycluster
import numpy as np
import scipy as sp
from scipy.stats import mode
from scipy.misc import factorial
import math

#from Bio import pairwise2
#from Bio.SubsMat import MatrixInfo as matlist
#from Bio.Seq import Seq
#from Bio.SeqRecord import SeqRecord
#from Bio.Alphabet import generic_protein
#from Bio import SeqIO


#MDS
def classicalmds(D,k,tol=1e-10):
    D = numpy.matrix(numpy.array(D)**2)
    n = D.shape[0]
    
    J = numpy.matrix(numpy.eye(n) - 1.0/n*numpy.ones((n,n)))
    
    B = -1.0/2*J*D*J
    
    L,Q = numpy.linalg.eigh(B)
    L[L < tol] = 0
    order = numpy.argsort(L)[::-1]
    se = numpy.square(L[order])
    Q = Q.take(order,1)[:,0:k]
    #print numpy.diag(L[order][0:k],0)
    L = numpy.sqrt(numpy.diag(L[order][0:k],0))
    X = Q*L
    return X, se


#Assuming that given problist starts from 1
def geometricapproximation(probmasslist):
    #check whether given array is prob distribution
    total=0.0
    for elem in probmasslist:
        total += elem
    assert total==1.0
    
    #make a cdf of probmasslist
    cdflist=[0.0]
    for index in range(0,len(probmasslist)):
        retvalue=probmasslist[index]
        cdfvalue=retvalue+cdflist[index]
        cdflist.append(cdfvalue)
    assert len(cdflist)==len(probmasslist)+1

    #estimate the new probabilities
    retlist=[]
    for index in range(0,len(probmasslist)):
        retvalue=probmasslist[index]
        newvalue=float(retvalue)/1.0-cdflist[index]
        retlist.append(newvalue)
    return retlist


#labelstrue and predicted are numpy arrays
def clusterevaluate(labelstrue,predicted,clustermetric):
    def combination(a,k):
        result=1
        for i in range(k):
            result=result*((a-i)/(k-i))
        return result
        
    def Precision(predicted,labels):
        K=np.unique(predicted)
        p=0
        for cls in K:
            cls_members=np.nonzero(predicted==cls)[0]
            if cls_members.shape[0]<=1:
               continue
            real_label=mode(labels[cls_members])[0][0]
            correctCount=np.nonzero(labels[cls_members]==real_label)[0].shape[0]
            p+=np.double(correctCount)/cls_members.shape[0]
        return p/K.shape[0]
        
    def Recall(predicted,labels):
        K=np.unique(predicted)
        ccount=0
        for cls in K:
            cls_members=np.nonzero(predicted==cls)[0]
            real_label=mode(labels[cls_members])[0][0]
            ccount+=np.nonzero(labels[cls_members]==real_label)[0].shape[0]
        return np.double(ccount)/predicted.shape[0]

    def f1score(predicted,labels):
        precision=Precision(predicted,labels)
        recall=Recall(predicted,labels)
        return (2.0*precision*recall)/(precision+recall),precision,recall
    
    #Adjusted purity
    #http://acl.ldc.upenn.edu/acl2002/MAIN/pdfs/Main303.pdf
    def APP(predicted,labels):
        K=np.unique(predicted)
        app=0.0
        for cls in K:
            cls_members=np.nonzero(predicted==cls)[0]
            if cls_members.shape[0]<=1:
               continue
            real_labels=labels[cls_members]
            correct_pairs=0
            for i in range(real_labels.shape[0]):
                for i2 in range(i+1):
                    if real_labels[i]==real_labels[i2]:
                       correct_pairs+=2
            total=cls_members.shape[0]
            total_pairs=total+1
            app+=np.double(correct_pairs)/(total_pairs*total)
        return np.double(app)/K.shape[0]

    #Mutual information
    def mutual_info(x,y):
        N=np.double(x.size)
        I=0.0
        eps = np.finfo(float).eps
        for l1 in np.unique(x):
            for l2 in np.unique(y):
                #Find the intersections
                l1_ids=np.nonzero(x==l1)[0]
                l2_ids=np.nonzero(y==l2)[0]
                pxy=(np.double(np.intersect1d(l1_ids,l2_ids).size)/N)+eps
                I+=pxy*np.log2(pxy/((l1_ids.size/N)*(l2_ids.size/N)))
        return I
    
    #Normalized mutual information
    def nmi(x,y):
        N=x.size
        I=mutual_info(x,y)
        Hx=0
        for l1 in np.unique(x):
            l1_count=np.nonzero(x==l1)[0].size
            Hx+=-(np.double(l1_count)/N)*np.log2(np.double(l1_count)/N)
        Hy=0
        for l2 in np.unique(y):
            l2_count=np.nonzero(y==l2)[0].size
            Hy+=-(np.double(l2_count)/N)*np.log2(np.double(l2_count)/N)
        return I/((Hx+Hy)/2)
    
    def grp2idx(labels):
        inds=dict()
        for label in labels:
            if label not in inds:
               inds[label]=len(inds)
        return np.array([inds[label] for label in labels])

    #Vmeasure
    def V(predicted,labels):
        predicted=grp2idx(predicted)
        labels=grp2idx(labels)
    
        a=np.zeros((np.unique(labels).size,np.unique(predicted).size))
        for i in range(a.shape[0]):
            for i2 in range(a.shape[1]):
                a[i,i2]=np.intersect1d(np.nonzero(labels==i)[0],np.nonzero(predicted==i2)[0]).size
        N=labels.size
        n=a.shape[0]
        a=np.double(a)
        Hck=0
        Hc=0
        Hkc=0
        Hk=0
        for i in range(a.shape[0]):
            for i2 in range(a.shape[1]):
                if a[i,i2]>0:
                   Hkc+=(a[i,i2]/N)*np.log(a[i,i2]/sum(a[i,:]))
                   Hck+=(a[i,i2]/N)*np.log(a[i,i2]/sum(a[:,i2]))
            Hc+=(sum(a[i,:])/N)*np.log(sum(a[i,:])/N)
        Hck=-Hck
        Hkc=-Hkc
        Hc=-Hc
        for i in range(a.shape[1]):
            ak=sum(a[:,i])
            Hk+=(ak/n)*np.log(ak/N)
        Hk=-Hk
    
        h=1-(Hck/Hc)
        c=1-(Hkc/Hc)
        vmeasure=(2*h*c)/(h+c)
        return vmeasure

    #Mutal information
    #http://acl.ldc.upenn.edu/acl2002/MAIN/pdfs/Main303.pdf
    def mi(predicted,labels):
        predicted=grp2idx(predicted)
        labels=grp2idx(labels)
        a=np.zeros((np.unique(labels).size,np.unique(predicted).size))
        for i in range(a.shape[0]):
            for i2 in range(a.shape[1]):
                a[i,i2]=np.intersect1d(np.nonzero(labels==i)[0],np.nonzero(predicted==i2)[0]).size   
        a=np.double(a)
        n=labels.size
        mi=0
        for i in range(a.shape[0]):
            for i2 in range(a.shape[1]): 
                if a[i,i2]>0:
                   mi+=a[i,i2]*np.log((a[i,i2]*n)/(sum(a[i,:])*sum(a[:,i2])))
        mi=mi/np.log(np.unique(labels).size*np.unique(predicted).size)
        mi=mi/n
        return mi

    #my Rand Score     
    def myrand(predicted, labels):
        acount=0
        bcount=0
        ccount=0
        dcount=0

        assert len(predicted)==len(labels)
        for index1 in range(0,len(predicted)):
            pre1=predicted[index1]
            label1=labels[index1]
            for index2 in range(index1+1,len(predicted)):
                pre2=predicted[index2]
                label2=labels[index2]
                if elem1==elem2 and label1==label2:
                   acount += 1
                elif elem1==elem2 and label1!=label2:
                   bcount += 1
                elif elem1!=elem2 and label1==label2:
                   ccount += 1
                elif elem1!=elem2 and label1!=label2:
                   dcount += 1

        randscore=(2.0*float(acount+dcount))/(len(predicted)*(len(predicted)-1))
        return randscore

    
    #Adjusted rand index
    #http://acl.ldc.upenn.edu/eacl2003/papers/main/p39.pdf
    def adjustedrand(predicted,labels):
        predicted=grp2idx(predicted)
        labels=grp2idx(labels)
        a=np.zeros((np.unique(labels).size,np.unique(predicted).size))
        for i in range(a.shape[0]):
            for i2 in range(a.shape[1]):
                a[i,i2]=np.intersect1d(np.nonzero(labels==i)[0],np.nonzero(predicted==i2)[0]).size
        cij=0
        a=np.double(a)
        for i in range(a.shape[0]):
            for i2 in range(a.shape[1]):
                if a[i,i2]>1:
                   cij+=combination(a[i,i2],2)
        ci=0
        for i in range(a.shape[0]):
            if sum(a[i,:])>1:
               ci+=combination(sum(a[i,:]),2)
        cj=0
        for i in range(a.shape[1]):
            if sum(a[:,i])>1:
               cj+=combination(sum(a[:,i]),2)
        cn=combination(double(labels.size),2)
        nominator=cij-((ci*cj)/cn)
        denominator=0.5*(ci+cj)-(ci*cj/cn)
        return nominator/denominator
    
    if clustermetric=="rand": #Rand index
         return myrand(predicted, labelstrue)         
    elif clustermetric=="adjustedrand": #Adjusted Rand index
         return adjustedrand(predicted, labelstrue)
    elif clustermetric=="fmeasure": #fmeasure
         return f1score(predicted,labelstrue)[0]
    elif clustermetric=="precision": #Precision
         return f1score(predicted,labelstrue)[1]
    elif clustermetric=="recall": #Recall
         return f1score(predicted,labelstrue)[2]
    elif clustermetric=="mi": #mutual information
         return mi(predicted,labelstrue)
    elif clustermetric=="nmi": #Normalized mutual information
         return nmi(predicted,labelstrue)
    elif clustermetric=="app": #adjusted purity
         return APP(predicted,labelstrue)  
    else:
        print "ERROR: Metric {0} is unknown!!".format(clustermetric)
        exit(1)


def katz_centrality(G,alpha=-1):
    rethash={}
    for node in G.nodes():
        rethash[node]=0.0
    if alpha==-1:    
       #estimate the maximum eigenvalue of matrix
       adjmat=nx.to_numpy_matrix(G)    
       [V,D]=sp.eigs(adjmat,1)
       maxeig=abs(D[0])
       matsize=adjmat.shape()[0]
       alpha=(1.0/maxeig)*0.9
    
    #estimate katz centrality
    retvals=(inv(eye(matsize)-(alpha*adjmat.transpose())) - eye(matsize))*ones(matsize,1)
    for myindex in range(0,len(retvals)):
        rethash[myindex]=retvals[myindex]
    return rethash

#my assortativity implemntation since networkx does not work!!
def assortativity(G):
    degrees=nx.degree(G)
    degrees_sq={}
    for node in degrees.keys():
        degrees_sq[node] = degrees[node]**2
 
    m = float(G.number_of_edges())
    num1, num2, den1 = 0, 0, 0
    for source, target in G.edges():
        num1 += degrees[source] * degrees[target]
        num2 += degrees[source] + degrees[target]
        den1 += degrees_sq[source] + degrees_sq[target]
 
    num1 /= m
    den1 /= 2*m
    num2 = (num2 / (2*m)) ** 2
 
    return (num1 - num2) / (den1 - num2)


diststats=["normalizedlaplacian","katzcentrality","degreecentrality","betweennesscentrality","eigenvectorcentrality","closenesscentrality","loadcentrality","currentflowclosenesscentrality","currentflowbetweennesscentrality","communicabilitycentrality","hubs","authorities"]
valuestats=["transitivity","diameter","radius","avg_shortest_path","average_square_clustering","assortativity","average_clustering"]
def returngraphstatistic(G,statisticname,eigencount=-1,mybin=[]):
   if statisticname in diststats:
    if statisticname=="normalizedlaplacian":
          normlapmat=nx.normalized_laplacian(G)
          if eigencount!=-1:
             kvalue=eigencount/len(mybin)
             for elem in mybin:
                 reteigvalues=sp.sparse.linalg.eigs(normlapmat, k=kvalue, sigma=elem+0.001)[0]
                 eigvalues=[]
                 for eigval in reteigvalues:
                     eig=sp.real(eigval)
                     eigvalues.append(eig)
                 distarray=sorted(eigvalues)
          else:
             reteigvalues=sp.linalg.eigvals(normlapmat)
             eigvalues=[]
             for eigval in reteigvalues:
                 eig=sp.real(eigval)
                 eigvalues.append(eig)
             distarray=sorted(eigvalues)
    elif statisticname=="katzcentrality":
       disthash=katz_centrality(G)
       distarray=sorted(disthash.values())
    elif statisticname=="degreecentrality":
       disthash=nx.degree_centrality(G)
       distarray=sorted(disthash.values())
    elif statisticname=="betweennesscentrality":
       disthash=nx.betweenness_centrality(G)
       distarray=sorted(disthash.values())
    elif statisticname=="eigenvectorcentrality":    
       #disthash=nx.eigenvector_centrality(G,max_iter=10000)
       disthash=nx.eigenvector_centrality_numpy(G)
       distarray=sorted(disthash.values())
    elif statisticname=="closenesscentrality":
       disthash=nx.closeness_centrality(G)
       distarray=sorted(disthash.values())
    elif statisticname=="loadcentrality":
       disthash=nx.load_centrality(G,normalized=True)
       distarray=sorted(disthash.values())
    elif statisticname=="currentflowclosenesscentrality":
       disthash=nx.current_flow_closeness_centrality(G,normalized=True)
       distarray=sorted(disthash.values())
    elif statisticname=="currentflowbetweennesscentrality":
       disthash=nx.current_flow_betweenness_centrality(G,normalized=True)
       distarray=sorted(disthash.values())
    elif statisticname=="communicabilitycentrality":
       disthash=nx.communicability_centrality_exp(G)
       distarray=sorted(disthash.values())
    elif statisticname=="hubs":
       hubhash=nx.hits(G,max_iter=10000)[0] 
       #hubhash=nx.hits_numpy(G)[0]
       distarray=sorted(hubhash.values())
    elif statisticname=="authorities":
       #authash=nx.hits_numpy(G)[1]
       authash=nx.hits(G,max_iter=10000)[1]
       distarray=sorted(authash.values())
    elif statisticname=="pagerank":
       pagehash=nx.pagerank(G,max_iter=10000)
       #pagehash=nx.pagerank_numpy(G,alpha=0.9)
       distarray=sorted(pagehash.values())
    elif statisticname=="closenessvitality":
       pagehash=nx.closeness_vitality(G)
       distarray=sorted(pagehash.values())
    elif statisticname=="clustering":
       pagehash=nx.clustering(G)
       distarray=sorted(pagehash.values())
    elif statisticname=="square_clustering":
       pagehash=nx.square_clustering(G)
       distarray=sorted(pagehash.values())
      
    hist=np.histogram(distarray, bins=100, range=(0,1), normed=True, weights=None, density=None)[0]
    return hist
   elif statisticname in valuestats:
    if statisticname=="modularity":
       return  
    elif statisticname=="transitivity":
       return nx.transitivity(G)
    elif statisticname=="average_clustering":
       return nx.average_clustering(G)
    elif statisticname=="avg_shortest_path":
       return nx.average_shortest_path_length(G)
    elif statisticname=="diameter":
       return nx.diameter(G)
    elif statisticname=="radius":
       return nx.radius(G)
    elif statisticname=="assortativity":
       return assortativity(G) #my implementation
       #return nx.degree_assortativity(G)
    elif statisticname=="average_square_clustering":
       total=0.0
       rethash=nx.square_clustering(G)
       for node in rethash.keys():
           total += rethash[node]
       return total/float(G.number_of_nodes()) 
   else:    
    print "ERROR: statistic {0} is UNKNOWN!!".format(statisticname)
    exit(1)
   

def convertsparse6(G,amtogpath):
    outfilename="sparse6"
    tempstr="".join(random.choice(string.ascii_uppercase + string.digits) for x in range(16))
    outfilename=tempstr+outfilename
    myfile="{0}myout".format(outfilename)
    file=open(myfile,"w")
    file.write("n={0}\n".format(G.number_of_nodes()))
    sortednodes=sorted(G.nodes())
    for node1 in sortednodes:
        rowstr=""
        for node2 in sortednodes:
            if G.has_edge(node1,node2):
               rowstr=rowstr+"1 "
            else:
               rowstr=rowstr+"0 " 
        file.write(rowstr+"\n")
    file.close()   

    code="{0} -s {1} {2}".format(amtogpath,myfile,outfilename)
    os.system(code)
    file=open(outfilename,"r")
    count=0
    for line in file:
        assert count<1
        line=line.rstrip()
        count +=1
    file.close()

    os.system("rm -rf {0}".format(myfile))
    os.system("rm -rf {0}".format(outfilename))
    return line

def convertgraph6(G,amtogpath):
    outfilename="graph6"
    tempstr="".join(random.choice(string.ascii_uppercase + string.digits) for x in range(16))
    outfilename=tempstr+outfilename
    myfile="{0}myout".format(outfilename)
    file=open(myfile,"w")
    file.write("n={0}\n".format(G.number_of_nodes()))
    sortednodes=sorted(G.nodes())
    for node1 in sortednodes:
        rowstr=""
        for node2 in sortednodes:
            if G.has_edge(node1,node2):
               rowstr=rowstr+"1 "
            else:
               rowstr=rowstr+"0 " 
        file.write(rowstr+"\n")
    file.close()   

    code="{0} {1} {2}".format(amtogpath,myfile,outfilename)
    os.system(code)
    file=open(outfilename,"r")
    count=0
    for line in file:
        assert count<1
        line=line.rstrip()
        count +=1
    file.close()
    
    os.system("rm -rf {0}".format(myfile))
    os.system("rm -rf {0}".format(outfilename))
    return line


def tablimitedfilereader(filename):
    domaindata = csv.reader(open(filename, "rb"), delimiter='\t')
    count=0
    nameshorizontal=[]
    namesvertical=[]
    data=[]
    for row in domaindata:
       if count==0:
          nameshorizontal=row[2:len(row)-1]
       else:
          namesvertical.append(row[0])
          cleanrow=[]
          for part in row[1:]:
             if part!="":
                cleanrow.append(part)      
          data.append(cleanrow)     
          
       count += 1

    if len((set(namesvertical)).difference(set(nameshorizontal)))!=0:
       print "ERROR:Horizontal and vertical names do not match!!"
       exit(1)

    #There are 2 columns with the same name!!
    if len(nameshorizontal)!=len(set(nameshorizontal)):
       for index1 in range(0,len(nameshorizontal)):
           for index2 in range(index1+1,len(nameshorizontal)):
              if nameshorizontal[index1]==nameshorizontal[index2]:
                 print "ERROR: There are 2 columns with the same name!!"
                 print nameshorizontal[index1]
                 print index1
                 print index2

    G=nx.Graph()
    for index1 in range(0,len(nameshorizontal)):
        name1=nameshorizontal[index1]
        G.add_node(name1)
        for index2 in range(0,len(nameshorizontal)):
           name2=nameshorizontal[index2]
           if data[index1][index2]==1:
               G.add_edge(name1,name2)
        
    return G
           
    #datahash={}
    #for index1 in range(0,len(nameshorizontal)):
    #    name1=nameshorizontal[index1]
    #    datahash[name1]={}
    #    for index2 in range(0,len(nameshorizontal)):
    #       name2=nameshorizontal[index2]
    #       datahash[name1][name2]=data[index1][index2]
   
    #return datahash
        

def makecumulative(temp):
    cdf=[]
    cdf.append(temp[0])
    for i in range(1,len(temp)):
        cdf[i]=cdf[i-1]+temp[i]
    return cdf
    

#emre 
def all_shortest_paths(G,a,b,len):

    if len==0:
        return []
    
    mylist=[]
    for elem in G[a]:
        path=nx.shortest_path(G,elem,b)
        if path!=False:
            val=nx.shortest_path_length(G,elem,b)
            if val==(len-1):
               mylist.append(elem)

    for elem in G[a]:
        for node in all_shortest_paths(G,elem,b,len-1):
           if node not in mylist:
              mylist.append(node)
              
    return mylist


def all_shortest_paths2(G,a,b): 
    """ Return a list of all shortest paths in graph G between nodes a  and b """ 
    ret = [] 
    pred = nx.predecessor(G,b) 
    if not pred.has_key(a):  # b is not reachable from a 
        return [] 
    pth = [[a,0]] 
    pthlength = 1  # instead of array shortening and appending, which  are relatively 
    ind = 0        # slow operations, we will just overwrite array  elements at position ind 
    while ind >= 0: 
        n,i = pth[ind] 
        if n == b: 
            ret.append(map(lambda x:x[0],pth[:ind+1])) 
        if len(pred[n]) > i: 
            ind += 1 
            if ind == pthlength: 
                pth.append([pred[n][i],0]) 
                pthlength += 1 
            else: 
                pth[ind] = [pred[n][i],0] 
        else: 
            ind -= 1 
            if ind >= 0: 
                pth[ind][1] += 1 
    return ret 


def findmaxpathlength(G):
    shortestpathmatrix=nx.all_pairs_shortest_path_length(G)
    
    maxpath=-1
    for in1 in range(0,len(sorted(G.nodes()))):
         for in2 in range(in1+1,len(sorted(G.nodes()))):
            node1=G.nodes()[in1]
            node2=G.nodes()[in2]
            curpath=int(shortestpathmatrix[node1][node2])
            if curpath > maxpath:
               maxpath=curpath

    if maxpath==-1:
       print "ERROR:This graph should be empty since no max path exists!!"
       exit(1)
    return maxpath
         

def functioncalculator(depth,Ontology):
    root=findroot(Ontology)     
    levelnodes=[]
    for node in Ontology.nodes():
       if node!=root: 
          val=nx.shortest_path_length(Ontology,root,node)
          if val==depth:
            levelnodes.append(node)
          if val==False:
             print "ERROR: This ontology is not connected!!"
             print node
             exit(1)
    return levelnodes


#depth is the level at which you want to come up with functions.
#The difference between this one and functioncalculator is implementation          
def functioncalculator2(depth,Ontology):
    root=findroot(Ontology) 
    leng=1
    temp=[]
    mylist=Ontology[root].keys()

    while(leng<depth):
      for node in mylist:
        for node2 in Ontology[node].keys():
          if node2 not in temp:
             temp.append(node2)

      mylist=[]       
      for node in temp:
        mylist.append(node)
   
      temp=[]
      leng=leng+1

    return mylist
      

def findroot(G):
    root=None
    for node in G.nodes():
        if len(G.predecessors(node))==0:   
           root=node
    if root==None:
       print "ERROR:There is no root in this graph"
       exit(1)
    return root 

#reads mips complex scheme file but ignores 550 part.
def mipscomplexontoreader(schemefilename,root="root"):
    complexonto=nx.DiGraph()
    id2names={}
    file=open(schemefilename,"r")
    for line in file:
        line=line.rstrip()
        if line=="" or line.startswith("X") or line.startswith("-"):
           continue
        parts=[]
        for part in line.split():
            if part!="":
               parts.append(part)
        id=parts[0]
        if not id.startswith("550"):
           name=" ".join(parts[1:])
           complexonto.add_node(id)
           id2names[id]=name
    file.close()    

    for index1 in range(0,len(complexonto.nodes())):
        node1=complexonto.nodes()[index1]
        parts1=node1.split(".")
        for index2 in range(index1+1,len(complexonto.nodes())):
            node2=complexonto.nodes()[index2]
            parts2=node2.split(".")
            if node1.startswith(node2):
               if len(parts1)-len(parts2)==1:
                  flag=True
                  for index in range(0,len(parts2)):
                      if parts1[index]!=parts2[index]:
                         flag=False 
                  if flag:
                     complexonto.add_edge(node2,node1)
            if node2.startswith(node1):
               if len(parts2)-len(parts1)==1:
                  flag=True
                  for index in range(0,len(parts1)):
                      if parts1[index]!=parts2[index]:
                         flag=False
                  if flag:
                     complexonto.add_edge(node1,node2)         

    for node in complexonto.nodes():
        if len(complexonto.predecessors(node))==0:
           complexonto.add_edge(root,node)
    complexonto.add_node(root)

    complexonto=complexonto.subgraph(nx.weakly_connected_components(complexonto)[0])
    for myid in id2names.keys():
        if myid not in complexonto.nodes():
           del id2names[myid]
    return (complexonto,id2names)


#This might have some errors. it is better to cal each subontolog returning functions seperately!!
def completeontoreader(filename):
    Ontology=nx.DiGraph()
    file=open(filename,'r')
    bioprocessannos=[]
    molecularannos=[]
    locationannos=[]
    Onto_name={}
    flag=False
    name=""
    for line in file:
        line=line.rstrip()
        if line=="[Term]":
           flag=True
           continue
        if flag:
           if line.startswith("id:"):
               splitted=line.split()
               node=int(splitted[1][3:])
               Ontology.add_node(node)
           elif line.startswith("name:"):
               splitted=line.split()
               name=""
               for elem in splitted[1:]:
                   name=name+elem+" "
               Onto_name[node]=name     
           elif line.startswith("namespace:"):
               splitted=line.split()
               if splitted[1]=="biological_process":
                   bioprocessannos.append(node)
               elif splitted[1]=="cellular_component":
                   locationannos.append(node)
               elif splitted[1]=="molecular_function":
                   molecularannos.append(node)
               else:
                   print "There is an error here!!"
                   exit(1)
           elif line.startswith("is_a:"):
               splitted=line.split()
               newnode=int(splitted[1][3:])
               Ontology.add_edge(newnode,node)
           elif line.startswith("relationship:"):
               pass
           elif not line:
               flag=False 
        #elif ( line.startswith("[Typedef]") and not flag): # For handling end of file
        #       break
    file.close()
    return Ontology

    #test part
    bio=nx.DiGraph.subgraph(Ontology, bioprocessannos )
    mol=nx.DiGraph.subgraph(Ontology, molecularannos )
    loc=nx.DiGraph.subgraph(Ontology, locationannos )
    print bio.number_of_nodes()
    print mol.number_of_nodes()
    print loc.number_of_nodes()
    print Ontology.number_of_nodes()
    print len(nx.weakly_connected_components(bio))
    print len(nx.weakly_connected_components(bio)[0])
    print "end"



    
def molecularontoreader(filename,connected,booleannames=False):
    Ontology=nx.DiGraph()
    file=open(filename,'r')
    bioprocessannos=set()
    molecularannos=set()
    locationannos=set()
    onto2name={}
    flag=False
    for line in file:
        line=line.rstrip()
        if line=="[Term]":
           flag=True
           continue
        if flag:
           if line.startswith("id:"):
               splitted=line.split()
               node=int(splitted[1][3:])
           elif line.startswith("name: "):
               name=line[6:]
               onto2name[node]=name
           elif line.startswith("namespace:"):
               splitted=line.split()
               if splitted[1]=="biological_process":
                   bioprocessannos.add(node)
               elif splitted[1]=="cellular_component":
                   locationannos.add(node)
               elif splitted[1]=="molecular_function":
                   molecularannos.add(node)
               else:
                   print "ERROR: Namespace {0} is UNKNOWN!!".format(splitted[1])
                   exit(1)
           elif line.startswith("is_a:"):
               splitted=line.split()
               newnode=int(splitted[1][3:])
               Ontology.add_edge(newnode,node)
           elif line.startswith("is_obsolete:"):
               splitted=line.split()
               if splitted[1]=="true":
                  del onto2name[node] 
               else:
                  print "ERROR: is_obsolete value {0} is UNKNOWN!!".format(splitted[1])
                  exit(1)
           elif line.startswith("relationship:"):
               pass
           elif line.startswith("intersection_of:"):
               pass     
           elif not line:
               flag=False 
        #elif ( line.startswith("[Typedef]") and not flag): # For handling end of file
        #       break
    file.close()

    Ontology=nx.DiGraph.subgraph(Ontology,molecularannos )
    if connected:
       Ontology=nx.weakly_connected_component_subgraphs(Ontology)[0]
       for onto in onto2name.keys():
           if onto not in Ontology.nodes():
              del onto2name[onto] 
              
    if booleannames:    
       return [Ontology,onto2name]
    else:
       return Ontology
    
def locationontoreader(filename,connected,booleannames=False):
    Ontology=nx.DiGraph()
    file=open(filename,'r')
    bioprocessannos=set()
    molecularannos=set()
    locationannos=set()
    onto2name={}
    flag=False
    for line in file:
        line=line.rstrip()
        if line=="[Term]":
           flag=True
           continue
        if flag:
           if line.startswith("id:"):
               splitted=line.split()
               node=int(splitted[1][3:])
           elif line.startswith("name: "):
               name=line[6:]
               onto2name[node]=name
           elif line.startswith("namespace:"):
               splitted=line.split()
               if splitted[1]=="biological_process":
                   bioprocessannos.add(node)
               elif splitted[1]=="cellular_component":
                   locationannos.add(node)
               elif splitted[1]=="molecular_function":
                   molecularannos.add(node)
               else:
                   print "ERROR: Namespace {0} is UNKNOWN!!".format(splitted[1])
                   exit(1)
           elif line.startswith("is_a:"):
               splitted=line.split()
               newnode=int(splitted[1][3:])
               Ontology.add_edge(newnode,node)
           elif line.startswith("is_obsolete:"):
               splitted=line.split()
               if splitted[1]=="true":
                  del onto2name[node] 
               else:
                  print "ERROR: is_obsolete value {0} is UNKNOWN!!".format(splitted[1])
                  exit(1)
           elif line.startswith("relationship:"):
               pass
           elif line.startswith("intersection_of:"):
               pass     
           elif not line:
               flag=False 
        #elif ( line.startswith("[Typedef]") and not flag): # For handling end of file
        #       break
    file.close()

    Ontology=nx.DiGraph.subgraph(Ontology,locationannos )
    if connected:
       Ontology=nx.weakly_connected_component_subgraphs(Ontology)[0]
       for onto in onto2name.keys():
           if onto not in Ontology.nodes():
              del onto2name[onto] 
              
    if booleannames:    
       return [Ontology,onto2name]
    else:
       return Ontology 

def bioprocessontoreader(filename,connected,booleannames=False):
    Ontology=nx.DiGraph()
    file=open(filename,'r')
    bioprocessannos=set()
    molecularannos=set()
    locationannos=set()
    onto2name={}
    flag=False
    for line in file:
        line=line.rstrip()
        if line=="[Term]":
           flag=True
           continue
        if flag:
           if line.startswith("id:"):
               splitted=line.split()
               node=int(splitted[1][3:])
           elif line.startswith("name: "):
               name=line[6:]
               onto2name[node]=name
           elif line.startswith("namespace:"):
               splitted=line.split()
               if splitted[1]=="biological_process":
                   bioprocessannos.add(node)
               elif splitted[1]=="cellular_component":
                   locationannos.add(node)
               elif splitted[1]=="molecular_function":
                   molecularannos.add(node)
               else:
                   print "ERROR: Namespace {0} is UNKNOWN!!".format(splitted[1])
                   exit(1)
           elif line.startswith("is_a:"):
               splitted=line.split()
               newnode=int(splitted[1][3:])
               Ontology.add_edge(newnode,node)
           elif line.startswith("is_obsolete:"):
               splitted=line.split()
               if splitted[1]=="true":
                  del onto2name[node] 
               else:
                  print "ERROR: is_obsolete value {0} is UNKNOWN!!".format(splitted[1])
                  exit(1)
           elif line.startswith("relationship:"):
               pass
           elif line.startswith("intersection_of:"):
               pass     
           elif not line:
               flag=False 
        #elif ( line.startswith("[Typedef]") and not flag): # For handling end of file
        #       break
    file.close()

    Ontology=nx.DiGraph.subgraph(Ontology,bioprocessannos )
    if connected:
       Ontology=nx.weakly_connected_component_subgraphs(Ontology)[0]
       for onto in onto2name.keys():
           if onto not in Ontology.nodes():
              del onto2name[onto] 
              
    if booleannames:    
       return [Ontology,onto2name]
    else:
       return Ontology 


#just for now. BE CAREFUL
#aliasseperator="{"

biogrid_genetical=[
"Synthetic Lethality",
"Phenotypic Suppression", 
"Phenotypic Enhancement",
"Dosage Rescue",
"Negative Genetic",
"Positive Genetic",
"Synthetic Growth Defect",
"Synthetic Rescue",
"Dosage Lethality", 
"Synthetic Haploinsufficiency",
"Dosage Growth Defect"]
biogrid_physical=[
"Affinity Capture-Luminescence",
"Affinity Capture-MS",
"Affinity Capture-RNA",
"Affinity Capture-Western",
"Biochemical Activity",
"Co-crystal Structure",
"Co-purification",
"Co-fractionation",
"Co-localization",
"Far Western",
"PCA",
"FRET",
"Protein-peptide",
"Protein-RNA",
"Reconstituted Complex",
"Two-hybrid"]
#returns the largest connected component of the graph
def biogridreadercombinednodename(filename,aliasseperator): 
    aliashash={}
    phy=nx.Graph()
    file=open(filename,"r")
    flag=False
    for line in file:
        line=line.rstrip()
        if line.startswith("INTERACTOR_A") and (not flag) :
            flag=True
            continue
        if flag: 
            splitted=line.split("\t")
            if (splitted[6] not in biogrid_physical) and (splitted[6] not in biogrid_genetical):
                print "ERROR: {0} experiment type is UNKNOWN!!".format(splitted[6])
                exit(1)
            if len(splitted)!=0 and (splitted[6] in biogrid_physical):
               first=splitted[2].lower()
               second=splitted[3].lower()
               if ( not aliashash.has_key(first) ):
                  aliashash[first]=set()
               if ( not aliashash.has_key(second) ):
                  aliashash[second]=set()

               alias1=set()
               alias1.add(splitted[0].lower())
               if splitted[4]!="N/A":
                  alias1 |= set(splitted[4].lower().split("|"))
  
               alias2=set()
               alias2.add(splitted[1].lower())
               if splitted[5]!="N/A":   
                  alias2 |= set(splitted[5].lower().split("|"))

               aliashash[first] |= alias1
               aliashash[second] |= alias2

               if first!=second:
                  if phy.has_edge(first,second):
                     phy[first][second]['weight'] += 1.0
                  else:
                     phy.add_edge(first,second,weight=1.0)
                     
    file.close() 

    #clean aliashash
    for node in aliashash.keys():
        if node in aliashash[node]:
           aliashash[node].remove(node)
           
    temp=nx.connected_component_subgraphs(phy)[0]
    phy=nx.Graph()
    for first,second,value in temp.edges(data=True):
        firstnode=str(first)
        for alias in aliashash[first]:
            firstnode=firstnode+aliasseperator+str(alias)
        secondnode=str(second)
        for alias in aliashash[second]:
            secondnode=secondnode+aliasseperator+str(alias)
        phy.add_edge(firstnode,secondnode,weight=value)

    return phy


def biogridreadergeneric(filename):
    aliashash={}
    phy=nx.Graph()
    file=open(filename,"r")
    flag=False
    for line in file:
        line=line.rstrip()
        if line.startswith("INTERACTOR_A") and (not flag) :
            flag=True
            continue
        if flag: 
            splitted=line.split("\t")
            if (splitted[6] not in biogrid_physical) and (splitted[6] not in biogrid_genetical):
                print "ERROR: {0} experiment type is UNKNOWN!!".format(splitted[6])
                exit(1)
            if len(splitted)!=0 and (splitted[6] in biogrid_physical):
               first=splitted[2].lower()
               second=splitted[3].lower()
               if ( not aliashash.has_key(first) ):
                  aliashash[first]=set()
               if ( not aliashash.has_key(second) ):
                  aliashash[second]=set()
               
               alias1=set()
               alias1.add(splitted[0].lower())
               if splitted[4]!="N/A":
                  alias1 |= set(splitted[4].lower().split("|"))
  
               alias2=set()
               alias2.add(splitted[1].lower())
               if splitted[5]!="N/A":   
                  alias2 |= set(splitted[5].lower().split("|"))

               aliashash[first] |= alias1
               aliashash[second] |= alias2

               if first!=second:
                  if phy.has_edge(first,second):
                     phy[first][second]['weight'] += 1.0
                  else:
                     phy.add_edge(first,second,weight=1.0)
                     
    file.close() 
    
    #clean aliashash
    for node in aliashash.keys():
        if node in aliashash[node]:
           aliashash[node].remove(node)
           
    phy=nx.connected_component_subgraphs(phy)[0]

    #removes the aliasases of nodes that are not in largest connected component
    for pro in aliashash.keys():
        if pro not in phy.nodes():
           del aliashash[pro]
    #testing  
    for node in phy.nodes():
        assert aliashash.has_key(node)
        
    return [phy,aliashash]
   

#ALWAYS returns a weighted graph
def biogridreader(filename,aliasseperator,outputtype):
    if outputtype=="combinednodename":
       phy=biogridreadercombinednodename(filename,aliasseperator)
       return phy
    elif outputtype=="generic":
       phy,aliashash=biogridreadergeneric(filename)
       return [phy,aliashash]
    else:
       print "error {0}".format(outputtype)
       exit(1)


def dipreader(filename): 
    aliashash={}
    phy=nx.Graph()
    file=open(filename,"r")
    flag=False
    for line in file:
        line=line.rstrip()
        if line.startswith("ID interactor") and (not flag) :
            flag=True
            continue
        if flag: 
            splitted=line.split("\t")
            if (splitted[6] not in physical) and (splitted[6] not in genetical):
                print "ERROR: {0} experiment type is UNKNOWN!!".format(splitted[6])
                exit(1)
            if len(splitted)!=0 and (splitted[6] in physical):
               first=splitted[2].lower()
               second=splitted[3].lower()
               if ( not aliashash.has_key(first) ):
                  aliashash[first]=set()
               if ( not aliashash.has_key(second) ):
                  aliashash[second]=set()

               alias1=set()
               alias1.add(splitted[0].lower())
               if splitted[4]!="N/A":
                  alias1 |= set(splitted[4].lower().split("|"))
  
               alias2=set()
               alias2.add(splitted[1].lower())
               if splitted[5]!="N/A":   
                  alias2 |= set(splitted[5].lower().split("|"))

               aliashash[first] |= alias1
               aliashash[second] |= alias2

               if first!=second:
                  if phy.has_edge(first,second):
                     phy[first][second]['weight'] += 1.0
                  else:
                     phy.add_edge(first,second,weight=1.0)
                     
    file.close() 

    #clean aliashash
    for node in aliashash.keys():
        if node in aliashash[node]:
           aliashash[node].remove(node)
           
    phy=nx.connected_component_subgraphs(phy)[0]

    #removes the aliasases of nodes that are not in largest connected component
    for pro in aliashash.keys():
        if pro not in phy.nodes():
           del aliashash[pro]
           
    #testing  
    for node in phy.nodes():
        assert aliashash.has_key(node)
        
    return [phy,aliashash]  


def readcommunityfile(filename,compressedcom,minsizelimit):
    if compressedcom:
        seennodes=set()
        nodes=[]
        realcommunities=[]
        file=open(filename,"r")
        cn=0
        edgeflag=False
        for line in file:
            line=line.rstrip()
            if cn==0:
               cn += 1 
               continue
            if line.startswith("#"):
               edgeflag=True
               continue
            if not edgeflag:
               index,name=line.split()
               nodes.append(name)
            else:
               parts=line.split("\t")
               outparts=[]
               if len(parts) >= minsizelimit:
                  for part in parts:
                      nodename=nodes[int(part)]
                      outparts.append(nodename)
                      seennodes.add(nodename) 
                  realcommunities.append(outparts)  
        file.close()

        return [realcommunities,seennodes]
    else:    
       seennodes=set()
       realcommunities=[]
       file=open(filename,"r")
       for line in file:
           line=line.rstrip()
           parts=line.split("\t")
           if len(parts) >= globals()["minsizelimit"]:
              for part in parts:
                  seennodes.add(part) 
              realcommunities.append(parts)         
       file.close()

       return [realcommunities,seennodes]



#NOT BEING USED RIGHT NOW
def writegraphmyline2(outpath,phy,linegraphseperator,weighted): 
    sortednodes=sorted(phy.nodes())
    indexphy=nx.Graph()
    for node1,node2,value in phy.edges(data=True):
        index1=-1
        index2=-1
        for index in range(0,len(sortednodes)):
            if sortednodes[index]==node1:
               index1=index
            if sortednodes[index]==node2:
               index2=index
            if index1!=-1 and index2!=-1:   
               break
        if index1==-1 or index2==-1:
           print "ERROR: node index has not been found!!"
           exit(1)
        indexphy.add_edge(index1,index2)
    linegraph=nx.line_graph(indexphy)
    outlinegraph=nx.Graph() 
    for edge in linegraph.edges():
        part1=sortednodes[int(edge[0][0])]
        part2=sortednodes[int(edge[0][1])]
        part3=sortednodes[int(edge[1][0])]
        part4=sortednodes[int(edge[1][1])]
        node1="{0}{1}{2}".format(part1,linegraphseperator,part2)
        node2="{0}{1}{2}".format(part3,linegraphseperator,part4)
        if weighted:
           set1=set([part1,part2])
           set2=set([part3,part4])
           val1=set1.difference(set2)
           val2=set2.difference(set1)
           neigh1=set(indexphy.neighbours(val1))
           neigh2=set(indexphy.neighbours(val2))
           value=float(len(neigh1.intersection(neigh2)))/len(neigh1.union(neigh2)) 
           outlinegraph.add_edge(node1,node2,weight=value)
        else:
           outlinegraph.add_edge(node1,node2) 
    if weighted:        
       graphwritemy(outpath,outlinegraph,True)
    else:
       graphwritemy(outpath,outlinegraph,False)

   
#Implemented to use LEAST MEMORY POSSIBLE.Memory error can be a problem for weighted versions of huge line graphs.
def writegraphmyline(outpath,phy,weighted): 
    sortednodes=sorted(phy.nodes())
    indexphy=nx.Graph()
    for node1,node2 in phy.edges_iter():
        index1=sortednodes.index(node1)
        index2=sortednodes.index(node2)
        indexphy.add_edge(index1,index2)
    file=open(outpath,"w")
    file.write("#index\n")
    for index in xrange(0,len(sortednodes)):
        file.write("{0} {1}\n".format(index,sortednodes[index]))
    file.close()    
    del sortednodes
    
    linegraph=nx.line_graph(indexphy)
    file=open(outpath,"a")
    file.write("#edgelist\n")
    if weighted:
       for edge in linegraph.edges_iter():
           set1=set(edge[0])
           set2=set(edge[1])
           val1=list(set1.difference(set2))
           assert len(val1)!=1
           val1=val1[0]
           val2=list(set2.difference(set1))
           assert len(val2)!=1
           val2=val2[0]
           neigh1=set(indexphy.neighbors(val1))
           neigh2=set(indexphy.neighbors(val2))
           value=float(len(neigh1.intersection(neigh2)))/len(neigh1.union(neigh2))
           file.write("{0} {1} {2}\n".format(edge[0],edge[1],value))
       file.close()    
    else:
       for line in nx.generate_edgelist(linegraph,data=['weight']):
           file.write("{0}\n".format(line))
       file.close()


#the case where compressed=False might not work properly right now!!
def readgraphmyline(linefilename,compressed,lineseperator="?"):
    nodes=[]
    index2node={}
    G=nx.Graph()
    file=open(linefilename,"r")
    cn=0
    edgeflag=False
    for line in file:
        line=line.rstrip()
        if cn==0:
           cn += 1  
           continue
        if line.startswith("#"):
           for index in xrange(0,len(index2node.keys())):
               nodes.append(index2node[index])
           edgeflag=True
           continue
        if not edgeflag:
           #find first blank spot and split accordingly 
           look=-1
           for index in xrange(0,len(line)):
               if line[index]==" ":
                  look=index+1
                  break
           index=int(line[0:look-1])
           name=line[look:]
           assert not index2node.has_key(index)
           if not index2node.has_key(index):
              index2node[index]=name
                          
            #parts=line.split()
            #index=int(parts[0])
            #name=parts[1]
            #assert not index2node.has_key(index)
            #index2node[index]=name
        else:
            parts=line.split()
            if compressed:
               node1=int(parts[0].replace("(","").replace(",",""))
               node2=int(parts[1].replace(")",""))
               node3=int(parts[2].replace("(","").replace(",",""))
               node4=int(parts[3].replace(")",""))
            else:
               node1=nodes[int(parts[0].replace("(","").replace(",",""))]
               node2=nodes[int(parts[1].replace(")",""))]
               node3=nodes[int(parts[2].replace("(","").replace(",",""))]
               node4=nodes[int(parts[3].replace(")",""))]
               
            newnode1="{0}{1}{2}".format(node1,lineseperator,node2)
            newnode2="{0}{1}{2}".format(node3,lineseperator,node4)
               
            if len(parts)==5:
               G.add_edge(newnode1,newnode2,weight=float(parts[4])) 
            elif len(parts)==4:
               G.add_edge(newnode1,newnode2)
            else:    
               print "error {0}".format(len(parts))
               exit(1)
    file.close()
    
    if compressed:
       return [G,nodes] 
    else:
       return G

#assign intersection score to G
def intersectionassigner(G):
    intscore=0.0
    for myindex1 in range(0,G.number_of_edges()):
        edge1=G.edges()[myindex1]
        for myindex2 in range(myindex1+1,G.number_of_edges()):
            edge2=G.edges()[myindex2]
            neigh1=G.neighbours(edge1)
            neigh2=G.neighbours(edge2)
            intscore += float(len(neigh1.intersection(neigh2)))/len(neigh1.union(neigh2)) 
    return intscore         


def unionassigner(G):
    unionscore=0.0
    for myindex1 in range(0,G.number_of_edges()):
        edge1=G.edges()[myindex1]
        for myindex2 in range(myindex1+1,G.number_of_edges()):
            edge2=G.edges()[myindex2]
            neigh1=G.neighbours(edge1)
            neigh2=G.neighbours(edge2)
            unionscore += float(len(neigh1.intersection(neigh2)))/len(neigh1.union(neigh2)) 
    return intscore         
 
def readaliasfile(aliasoutpath):
    aliashash={}
    file=open(aliasoutpath,"r")
    for line in file:
        line=line.rstrip()
        parts=line.split("\t")
        pro=parts[0]
        values=parts[1:]
        assert not aliashash.has_key(pro)
        temp=set()
        temp|=set(values)
        aliashash[pro]=temp
    file.close()
    return aliashash
        
#If no alias exists, do not put that node in the file
def writealiasfile(outpath,aliashash):
    file=open(outpath,"w")
    for pro in aliashash.keys():
        values=[pro]
        values.extend(aliashash[pro])
        if len(values)>1:
           temp="\t".join(values)
           file.write(temp+"\n")
    file.close()


#write graphs in my format!!    
def writegraphmy(outpath,phy,weighted):
    sortednodes=sorted(phy.nodes())
    file=open(outpath,"w")
    file.write("#index\n")
    for index in xrange(0,len(sortednodes)):
        file.write("{0} {1}\n".format(index,sortednodes[index]))
    file.write("#edgelist\n")
    indexphy=nx.Graph()
    for node1,node2,value in phy.edges_iter(data=True):
        index1=sortednodes.index(node1)
        index2=sortednodes.index(node2)
        if weighted:    
           indexphy.add_edge(index1,index2,weight=value['weight'])
        else:
           indexphy.add_edge(index1,index2) 
    
    #output graph that is made of indexes to file
    for line in nx.generate_edgelist(indexphy,data=['weight']):
        file.write("{0}\n".format(line))    
    file.close()

    
#read graphs that are in my format!!
def readgraphmy(outpath,compressed):
    nodes=[]
    index2node={}
    G=nx.Graph()
    file=open(outpath,"r")
    cn=0
    edgeflag=False
    for line in file:
        line=line.rstrip()
        if cn==0:
           cn += 1  
           continue
        if line.startswith("#"):
           for index in xrange(0,len(index2node.keys())):
               nodes.append(index2node[index])
           edgeflag=True
           continue
        if not edgeflag:
           #find first blank spot and split accordingly 
           look=-1
           for index in xrange(0,len(line)):
               if line[index]==" ":
                  look=index+1
                  break
           index=int(line[0:look-1])
           name=line[look:]
           assert not index2node.has_key(index)
           if not index2node.has_key(index):
              index2node[index]=name
        else:
            parts=line.split()
            if compressed:
               node1=int(parts[0])
               node2=int(parts[1])
            else:    
               node1=nodes[int(parts[0])]
               node2=nodes[int(parts[1])]
            if len(parts)==2:
                G.add_edge(node1,node2)
            elif len(parts)==3:
                G.add_edge(node1,node2,weight=float(parts[2]))   
    file.close()
    
    if compressed:
       return [G,nodes] 
    else:
       return G


  
def write2files(combinedphy,filename,aliasout=True):
    phy=nx.Graph()
    for edge in combinedphy.edges():
        n1=edge[0].split(aliasseperator)[0]
        n2=edge[1].split(aliasseperator)[0]
        phy.add_edge(n1,n2)
                          
    outfile=filename.replace("BIOGRID-ORGANISM-","").replace("-3.0.65.tab","")
    nx.write_edgelist(phy,outfile)

    if aliasout:
        aliashash={}
        for node in combinedphy.nodes():
            node=node.split(aliasseperator)[0]
            alias=node.split(aliasseperator)[1:]
            if not aliashash.has_key(node):
               aliashash[node]=set()
               for elem in alias:
                  aliashash[node].add(elem)
            else:
               for elem in alias:
                  aliashash[node].add(elem)
                  
        outfile=filename.replace("BIOGRID-ORGANISM-","").replace("-3.0.65.tab","").replace(".txt","")+"-alias.txt"
        file=open(outfile,'w')
        for key in aliashash.keys():
            if len(aliashash[key])>0:
               aliasstring=""
               for elem in aliashash[key]:
                  aliasstring=aliasstring+delimiter+elem
            file.write(str(key)+aliasstring+"\n")               
        file.close()


       
def plainppireader(filename,delimiter="\t"):
    G=nx.Graph()
    file=open(filename)
    for line in file:
        protein=line.rstrip().split(delimiter)
        G.add_edge(protein[0],protein[1])
    file.close()
     
    aliashash={}
    filename=filename.replace(".txt","")+"-alias.txt"
    file=open(filename)
    for line in file:
       mylist=line.rstrip().split(delimiter)
       key=mylist[0]
       mylist.remove(key)
       aliashash[key]=mylist
    file.close()
    
    return [G,aliashash]



def myrange(start, stop, step):
     r = start
     retlist=[]
     while r < stop:
         retlist.append(r)
         r += step
         
     return retlist


def listdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name)) and (not name.startswith(".DS_Store"))]

def listfiles(dir):
    """list files
    """
    return [d for d in os.listdir(dir) if not os.path.isdir(d) and (not d.startswith(".DS_Store")) and (not d.startswith("._.DS_Store"))]


main_evidence_codes=["EXP","IDA","IPI","IMP","IGI","IEP"]
comp_evidence_codes=["ISS","ISO","ISA","ISM","IGC","IBA","IBD","IKR","IRD","RCA"]
other_evidence_codes=["TAS","NAS","IC","ND"]
other_evidence_codes2=["IEA","NR"]
def annocleaner(filename,outprefix):
    file=open(filename,"r")
    outfilename="{0}{1}".format(outprefix,filename)
    outfile=open(outfilename,"w")
    exist=0
    nonexist=0
    for line in file:
        if not line.startswith("!"):
           parts=line.rstrip().split("\t")
           if parts[6] in main_evidence_codes+comp_evidence_codes+other_evidence_codes+other_evidence_codes2:
              if parts[6] in main_evidence_codes+comp_evidence_codes+other_evidence_codes:
                 outfile.write(line)
                 exist +=1
              else:
                 nonexist +=1 
           else:
              print "Annocode {0} is UNKNOWN!!".format(parts[6])
              exit(1)         
    file.close()
    outfile.close()
    #print "Annocleaner"
    #print "exist {0}".format(exist)
    #print "nonexist {0}".format(nonexist)
    #os.system("rm -rf {0}".format(filename))
    #os.system("mv {0} {1}".format(outfilename,filename))


#remove the annotations from allannotations which has less count than atleastnumber(allannoations list can also be a subset of annotations seen in pro2anno)
def preprocessanno(pro2anno,allannotations,atleastnumber):
    if allannotations=="-1":
       allannotations=[item for sublist in pro2anno.values() for item in sublist]

    deleteanno=[]
    for anno in allannotations:
        seencount=0 
        for pro in pro2anno.keys():
            if anno in pro2anno[pro]:
               seencount += 1
        if seencount < atleastnumber:
           deleteanno.append(anno)

    for delanno in deleteanno:
        for pro in pro2anno.keys():
            if delanno in pro2anno[pro]:
               pro2anno[pro].remove(delanno)
                
    #clean proteins(keys) that has no more annotations left
    for pro in pro2anno.keys():
        if len(pro2anno[pro])==0:
           del pro2anno[pro]
            
    #allannotations=[anno for anno in allannotations if anno not in deleteanno]
    return pro2anno 


#We estimate the overlap ratio by only considering communities.We don't consider the proteins that are not seen in communities since we only care about how community assigned portion overlaps not what portion of the network is assigned community.
def estimateoverlapmetric(communities,metrictype):
    if metrictype=="overlap":
       norm=len(communities)*(len(communities)-1)/2.0
       total=0.0
       for index1 in range(0,len(communities)):
           com1=communities[index1]
           for index2 in range(index1+1,len(communities)): 
               com2=communities[index2]
               maxcom=float(max([len(com1),len(com2)]))
               intercom=float(len(com1.intersection(com2)))
               total += intercom/maxcom
       return float(total)/norm
    elif metrictype=="overlap_union":
       norm=len(communities)*(len(communities)-1)/2.0
       total=0.0
       for index1 in range(0,len(communities)):
           com1=communities[index1]
           for index2 in range(index1+1,len(communities)): 
               com2=communities[index2]
               intercom=float(len(com1.intersection(com2)))
               unioncom=float(len(com1.union(com2)))
               total += intercom/unioncon
       return float(total)/norm
    elif metrictype=="my_union":
       norm=0.5*(len(communities)*(len(communities)-1)/2.0)
       total=0.0
       for index1 in range(0,len(communities)):
           com1=communities[index1]
           for index2 in range(index1+1,len(communities)): 
               com2=communities[index2]
               intercom=float(len(com1.intersection(com2)))
               unioncom=float(len(com1.union(com2)))
               ratio=intercom/unioncom 
               total += min(ratio,1.0-ratio)
       return float(total)/norm        
    elif metrictype=="my":
       norm=0.5*(len(communities)*(len(communities)-1)/2.0)
       total=0.0
       for index1 in range(0,len(communities)):
           com1=communities[index1]
           for index2 in range(index1+1,len(communities)): 
               com2=communities[index2]
               intercom=float(len(com1.intersection(com2)))
               maxcom=float(max([len(com1),len(com2)]))
               ratio=intercom/maxcom 
               total += min(ratio,1.0-ratio)
       return float(total)/norm 
    else:
       print "Overlap metric {0} is UNKNOWN!!!".format(metrictype)
       exit(1)



def annofilewriter(filename,pro2function,G,indices="-1"):
    if indices=="-1":
       file=open(filename,"w")
       for pro in pro2function.keys():
           if pro in G.nodes():
              outstr=str(pro)
              if len(pro2function[pro])!=0:
                 for func in pro2function[pro]:
                     outstr=outstr+"\t"+str(func)    
                 file.write(outstr+"\n")
       file.close()
    else: #compressed so indices matter
       file=open(filename,"w")
       for pro in pro2function.keys():
           if pro in G.nodes():
              proname=indices[pro] 
              outstr=str(proname)
              if len(pro2function[pro])!=0:
                 for func in pro2function[pro]:
                     outstr=outstr+"\t"+str(func)    
                 file.write(outstr+"\n")
       file.close()   




#GO numbers should be INTEGER.THAT IS IMPORTANT
#if all charactes of annotation is number, then return int
#otherwise annotations are strings       
def  annofilereader(filename,minsizelimit=None,annotype="str"):
     #READ ANNO FILE, Proteinname \t ANNOs
     pro2anno={}
     allannotations=set()
     file=open(filename,"r")
     for line in file:
         parts=line.rstrip().split("\t")
         if len(parts)<2:
            print "Error!!There is no annotation for protein {0}".format(parts[0])
            exit(1)
         pro=parts[0]
         annos=parts[1:]
         if annotype!="str":    
            typedannos=[]
            for anno in annos:
                if annotype=="int":
                   typedannos.append(int(anno))
                elif annotype=="float":
                   typedannos.append(float(anno))
                else:
                   print "This annotype format {0} is unknown!!".format(annotype)
                   exit(1)
            allannotations|=set(typedannos)    
            if pro2anno.has_key(pro):
               pro2anno[pro]|=set(typedannos)     
            else:
               pro2anno[pro]=set()
               pro2anno[pro]|=set(typedannos)
         else:
            allannotations|=set(annos)    
            if pro2anno.has_key(pro):
               pro2anno[pro]|=set(annos)     
            else:
               pro2anno[pro]=set()
               pro2anno[pro]|=set(annos)
     file.close()

     if minsizelimit!=None:
         removeannos=set()
         for anno in allannotations:
             seencount=0
             for pro in pro2anno.keys():
                 if anno in pro2anno[pro]:
                     seencount +=1
             if seencount<minsizelimit:
                 removeannos.add(anno) 
                 for pro in pro2anno.keys():
                     if anno in pro2anno[pro]: 
                         pro2anno[pro].remove(anno)

         allannotations -= removeannos
         for pro in pro2anno.keys():
             if len(pro2anno[pro])==0:
                del pro2anno[pro]
                    
     return [pro2anno,allannotations]


# Given a node and basenodes, find all upper nodes of a given node in a graph
# Given a node and basenodes, that perfectly defines a direction on a tree so we can find upper nodes on that tree. 
def upperpartnodesfinder(G,node,basenodes):
    mylist=[node]
    if node in basenodes:
       return mylist
    suclist=G.successors(node)
    if len(suclist)==0:
       return mylist
    
    set1=set(upperpartnodesfinder(G,suclist[0],basenodes))
    set2=set(upperpartnodesfinder(G,suclist[1],basenodes))
    set3=set(mylist)
    
    temp=set1.union(set2)
    return list(temp.union(set3))


#Find all leave nodes under a node in a dag G
def subleavesfinder(node,G):
    suclist=G.successors(node)
    if len(suclist)==0:
       retset=set()
       retset.add(node)
       return retset
    else:
       unionset=set()
       for suc in suclist: 
           unionset |= subleavesfinder(suc,G)
       return unionset
 
#Find all nodes(not just leaves) under a node in a dag G
def suballnodesfinder(node,G):
    suclist=G.successors(node)
    unionset=set()
    unionset.add(node)
    for suc in suclist: 
        unionset |= suballnodesfinder(suc,G)
    return unionset

    
#assign edge to each ancestor-child relationships 
def tree2dag(G):
    retG=nx.Graph(G)
    for node in retG.nodes():
        subnodes=suballnodesfinder(node,G)
        for subnode in subnodes:
            retG.add_edge(node,subnode)
    return retG        
#Similar to tree2dag but input graph is also a DAG    
def dag2dag(G):
    retG=nx.Graph(G)
    for node in retG.nodes():
        subnodes=suballnodesfinder(node,G)
        for subnode in subnodes:
            retG.add_edge(node,subnode)
    return retG
            

def writevicutanno(pro2anno,vicutannofilename):
    file=open(vicutannofilename,"w")
    for pro in pro2anno.keys():
        for val in pro2anno[pro]:
            file.write("{0}\t{1}\n".format(pro,val))
    file.close()
    

 
def modifyslimfile(slimfile,outslimfile,annotations):
    file=open(slimfile,'r')
    outfile=open(outslimfile,'w')
    bioprocessannos=set()
    molecularannos=set()
    locationannos=set()
    flag=False
    for line in file:
        line=line.rstrip()
        if line=="[Term]":
           flag=True
           templines=""
           templines="{0}{1}{2}".format(templines,line,"\n")
           continue
        if flag:
           templines="{0}{1}{2}".format(templines,line,"\n") 
           if line.startswith("id:"):
               splitted=line.split()
               node=int(splitted[1][3:])
           elif line.startswith("name: "):
               pass
           elif line.startswith("namespace:"):
               splitted=line.split()
               if splitted[1]=="biological_process":
                   bioprocessannos.add(node)
               elif splitted[1]=="cellular_component":
                   locationannos.add(node)
               elif splitted[1]=="molecular_function":
                   molecularannos.add(node)
               else:
                   print "ERROR: Namespace {0} is UNKNOWN!!".format(splitted[1])
                   exit(1)
           elif line.startswith("is_a:"):
               pass
           elif line.startswith("is_obsolete:"):
               pass
           elif line.startswith("relationship:"):
               pass
           elif line.startswith("intersection_of:"):
               pass     
           elif not line:
               flag=False
               if node in annotations:
                  outfile.write(templines)
        #elif ( line.startswith("[Typedef]") and not flag): # For handling end of file
        #       break
    file.close()
    outfile.close()
      

def readslimfile(filename,annotype):
    file=open(filename,'r')
    bioprocessannos=set()
    molecularannos=set()
    locationannos=set()
    flag=False
    for line in file:
        line=line.rstrip()
        if line=="[Term]":
           flag=True
           continue
        if flag:
           if line.startswith("id:"):
               splitted=line.split()
               node=int(splitted[1][3:])
           elif line.startswith("name: "):
               pass
           elif line.startswith("namespace:"):
               splitted=line.split()
               if splitted[1]=="biological_process":
                   bioprocessannos.add(node)
               elif splitted[1]=="cellular_component":
                   locationannos.add(node)
               elif splitted[1]=="molecular_function":
                   molecularannos.add(node)
               else:
                   print "ERROR: Namespace {0} is UNKNOWN!!".format(splitted[1])
                   exit(1)
           elif line.startswith("is_a:"):
               pass
           elif line.startswith("is_obsolete:"):
               pass
           elif line.startswith("relationship:"):
               pass
           elif line.startswith("intersection_of:"):
               pass     
           elif not line:
               flag=False 
        #elif ( line.startswith("[Typedef]") and not flag): # For handling end of file
        #       break
    file.close()

    allset=set()
    if annotype=="bioprocessontoreader":
       allset|=bioprocessannos 
    elif annotype=="molecularontoreader":
       allset|=molecularannos 
    elif annotype=="locationontoreader":
       allset|=locationannos
    else:
       print "ERROR: this annotype{0} is unknown!!".format(annotype) 
    return allset
                

def readcomplexannotations(annofilename):
    pro2function={}
    file=open(annofilename,"r")
    for line in file:
        line=line.rstrip()
        parts=line.split("\t")
        nodename=parts[0]
        annos=set(list(parts[1:]))
        assert not pro2function.has_key(nodename)
        pro2function[nodename]=annos
    file.close()
    return pro2function
    
#currently not just reads but also find the corresponding function at that depth(detailed=true)
def readGOannotations(filename,specificanno,annotations,aliasseperator,Ontology="-1"):
    if Ontology=="-1":
       detailed=False
    else:
       detailed=True

    #linecount=0   
    temppro2function={}
    aliashash={}
    file=open(filename,"r")
    for line in file:
        #print "linecount: {0}".format(linecount)
        #linecount += 1
        line=line.rstrip()
        if line.startswith("!"): 
           continue
        parts=line.split("\t")
        uniquename=parts[1].lower()
        gene=parts[2].lower()

        if ( not aliashash.has_key(gene) ):
            aliashash[gene]=set()
        alias=set()
        alias.add(uniquename)
        if len(parts[10])!=0:
           alias |= set(parts[10].lower().split("|"))
        aliashash[gene] |= alias

        if ( parts[3][0:2]=="GO"): 
           onto=int(parts[3][3:])
        else:
           onto=int(parts[4][3:])

        if specificanno:
           ontoset=set() 
           if detailed: #Ontology must also be given which is nx digraph
              for anno in annotations:
                  if onto in Ontology.nodes(): #since some annotations won't be part of largest connected component of Ontology
                     if Ontology.has_edge(anno,onto):
                        ontoset.add(anno) 
           else:
              if onto in annotations:
                 ontoset.add(onto)

           if len(ontoset)!=0:
              if temppro2function.has_key(gene):
                 temppro2function[gene]|=ontoset
              else:
                 temppro2function[gene]=ontoset  
        else:
           ontoset=set()
           ontoset.add(onto)  
           if temppro2function.has_key(gene):
              temppro2function[gene]|=ontoset
           else:
              temppro2function[gene]=ontoset
    file.close()

    #clean aliashash
    for node in aliashash.keys():
        if node in aliashash[node]:
           aliashash[node].remove(node)
           
    pro2function={}
    for node in temppro2function.keys():
        nodename=str(node)
        for alias in aliashash[node]:
            nodename=nodename+aliasseperator+str(alias)
        pro2function[nodename]=temppro2function[node]
    
    return pro2function


def complexontoreader(schemefilename,root="root"):
    complexonto=nx.DiGraph()
    file=open(schemefilename,"r")
    for line in file:
        line=line.rstrip()
        if line=="":
           continue
        if line.startswith("X") or line.startswith("-"):
           continue
        parts=[]
        for part in line.split():
            if part!="":
               parts.append(part)
        id=parts[0]
        name=" ".join(parts[1:])
        complexonto.add_node(id) 
    file.close()    

    for index1 in range(0,len(complexonto.nodes())):
        node1=complexonto.nodes()[index1]
        parts1=node1.split(".")
        for index2 in range(index1+1,len(complexonto.nodes())):
            node2=complexonto.nodes()[index2]
            parts2=node2.split(".")
            if node1.startswith(node2):
               if len(parts1)-len(parts2)==1:
                  flag=True
                  for index in range(0,len(parts2)):
                      if parts1[index]!=parts2[index]:
                         flag=False 
                  if flag:
                     complexonto.add_edge(node2,node1)
            if node2.startswith(node1):
               if len(parts2)-len(parts1)==1:
                  flag=True
                  for index in range(0,len(parts1)):
                      if parts1[index]!=parts2[index]:
                         flag=False
                  if flag:
                     complexonto.add_edge(node1,node2)         

    for node in complexonto.nodes():
        if len(complexonto.predecessors(node))==0:
           complexonto.add_edge(root,node)
    complexonto.add_node(root)

    complexonto=complexonto.subgraph(nx.weakly_connected_components(complexonto)[0])
    return complexonto


#Write estimations to file
def estimate2file(estimated,filename):
    file=open(filename,"w")
    for pro in estimated.keys():
        temp=str(pro)
        for fonk in estimated[pro]:
            temp=temp+"\t"+str(fonk)
        temp=temp+"\n"
        file.write(temp)    
    file.close()


#I think we don't need this IN THE FUTURE!!
def setcombiner(multisets):
    retset=set()
    for multiset in multisets:
        retset.union(multiset)
    return retset
   

def invert_dict(indict):
    outdict = {}
    for mykey in indict.keys():
        if type(indict[mykey])==set or type(indict[mykey])==list:
            #print "set"
           for myvalue in indict[mykey]:
               if outdict.has_key(myvalue):
                  outdict[myvalue].add(mykey)   
               else:
                  outdict[myvalue]=set()
                  outdict[myvalue].add(mykey)
        else:
        #if type(indict[mykey])==str:
           myvalue=indict[mykey]
           if outdict.has_key(myvalue):
              outdict[myvalue].add(mykey)   
           else:
              outdict[myvalue]=set()
              outdict[myvalue].add(mykey)

    return outdict


def hier2vicuttree(hier,treefilename,nodenum):
    file=open(treefilename,"w")
    internalnode=-1
    for elem in hier:
        node1=int(elem[0])
        node2=int(elem[1])
        if node1>=nodenum:
           node1 = (nodenum-node1)-1
        if node2>=nodenum:
           node2 = (nodenum-node2)-1
        file.write("{0}\t{1}\t{2}\n".format(node1,node2,internalnode))
        internalnode -= 1
    file.close()

    assert internalnode==(-1*nodenum) 


#==============================================================================
# Node similarity functions.
#==============================================================================

def jaccard_matrix(G,numberincides=True):
    A = np.zeros((G.order(),G.order()),dtype=np.float)
    if numberincides:
       for node1 in xrange(0,G.order()):
           for node2 in xrange(node1+1,G.order()):
               A[node1][node2] = (1-jaccard(G,node1,node2)) # Distance, not similarity!!
               A[node2][node1] = A[node1][node2]
    else:
        ids2nodes = {}
        for i,u in enumerate(G.nodes()):
            ids2nodes[i] = u
            for j,v in enumerate(G.nodes()):
                A[i][j] = (1-jaccard(G,u,v)) # Distance, not similarity!!

    #logging.info("Checking if A matrix is symmetric.")
    for i in xrange(G.order()):
        for j in xrange(G.order()):
            assert A[i][j] == A[j][i]
    if numberincides:
       return A
    else:
       return (A,ids2nodes)

def jaccard(G,u,v):
    """ Computes the jaccard coefficient of u and v (could be supernodes).
        J(u,v) = #mutual neighbors / # total neighbors.
    """
    x = set(G.neighbors(u))
    y = set(G.neighbors(v))
    return float(len(x.intersection(y))) / len(x.union(y))


def dice_matrix(G,numberincides=True):
    A = np.zeros((G.order(),G.order()),dtype=np.float)
    if numberincides:
       for node1 in xrange(0,G.order()):
           for node2 in xrange(node1+1,G.order()):
               A[node1][node2] = (1-dice(G,node1,node2)) # Distance, not similarity!!
               A[node2][node1] = A[node1][node2]
    else:
        ids2nodes = {}
        for i,u in enumerate(G.nodes()):
            ids2nodes[i] = u
            for j,v in enumerate(G.nodes()):
                A[i][j] = (1-dice(G,u,v)) # Distance, not similarity!!

    #logging.info("Checking if A matrix is symmetric.")
    for i in xrange(G.order()):
        for j in xrange(G.order()):
            assert A[i][j] == A[j][i]
    if numberincides:
       return A
    else:
       return (A,ids2nodes)
        
def dice(G,u,v):
    """ Computes the dice coefficient for two nodes (parameter to include
        self).
    """
    x = set(G.neighbors(u))
    y = set(G.neighbors(v))
    x.add(u) # Self-loop.
    y.add(v) # Self-loop.
    return float(2.0*len(x.intersection(y))) / (len(x) + len(y))


def process_kmeans(ids2nodes,clusterids):
    """ Reads the k-means output into nodes2modules,modules2nodes. """
    m2n = {}
    for i,u in enumerate(clusterids):
        m2n.setdefault(u,set([])).add(ids2nodes[i])
    return (invert_dict(m2n),m2n)
    
def kmeans(A,ids2nodes,nc):
    """ Runs k-means using the Pycluster package for the given number of
        clusters.
    """
    clusterids,error,nfound = Pycluster.kcluster(data=A,npass=100,dist='c',nclusters=nc)
    #clusterids = rpy.r.kmeans(A,nc)["cluster"]
    assert len(clusterids) == len(A)
    (nodes2modules,modules2nodes) = process_kmeans(ids2nodes,clusterids)
    return (nodes2modules,modules2nodes)


    
#??? has not been implemented correctly
def ppimapreader(filename):
     #READ MAP FILE, Proteinnum \t  ORIGINAL proteinanme
     count2pro={}
     file=open(filename,"r")
     parts=line.rstrip().split("\t")
     if len(parts)!=2:
           print "error"
           print len(parts)
           exit(1)
     count2pro[parts[0]]=parts[1]      

     file.close()
     return count2pro




 
def nodeswithindistance(G,startnode,radius):
    if radius<1:
       return set()
    elif radius==1:
       return set(G.neighbors(startnode))
    else:
       retlist=set()
       propath=nx.single_source_shortest_path_length(G,startnode)
       for node in propath.keys():
           if propath[node] <= radius:
              retlist.add(node)
       return retlist

            
#Returns all nodes in the given radius of given node
"""def nodeswithindistance(G,protein,examined,radius):
    if radius<1:
        return set()
    elif radius==1:
        neighlist=set(G.neighbors(protein))
        return neighlist-examined
    else:
        neighlist=G.neighbors(protein)
        for elem in neighlist:
            examined.add(elem)
        for elem in neighlist:
            if elem in examined:
               neightlist.remove(elem)
               
        retlist=set()
        for elem in neighlist:
            for elem2 in neighprotein(G,elem,examined,radius-1)
                retlist.add(elem2)
                
        return retlist"""
