PROBC: PREDICTING HI-C INTERACTIONS FROM EPIGENETIC AND TRANSCRIPTION MODIFICATIONS

There are two sample files in this directory for training:
train_order1_width2_preloglinear_nonparam_lambda3800.0_iter100_dommetadata.txt
train_order1_width2_preloglinear_nonparam_lambda3800.0_iter100_domparams.txt

Proposed method ProbC has two parts:

1- estCRFParamsNonparam.py: estimates the model parameters. It has the following input parameters:

markerpath: marker files for corresponding domain partitions. It is a
combination of marker data of different chromosomes. Marker data of
each chromosome is separated by blank.

The format is as follows:
marker \t bin \t value
....

where each line defines the value of a marker at a given bin.

domainpath: domain file. The format is as follows:
nodecount\t1000
start,end\m
....

where start and end are the beginning and end of domains~(including
both of them). First line says that we have 1000 bins. It is a
combination of domain partition of different chromosomes. Domain partition of
each chromosome is separated by blank.

outprefix: prefix of output files. It generates two output
files. Metadata file: which includes information about the procedure,
and domparams file: which includes the estimated model parameters.

lambdaval: regularization lambda for smoothness

grlambdaval: group lambda value

width: width of each bin on affecting TADs. We always use 1. 

prepromodel: model for preprocessing input marker data (such as log transformation).

itercount: Maximum number of iterations of the optimization procedure.

basecount: number of kernel bases.

cb, ci, ce: relative weights of boundaries, interior and external segments.


2- findDomain.py: estimates TADs by using model parameters and markers.

markerpath: marker file similar to above. However, it has markers of
a single chromosome to be predicted.

parampath: model parameters estimated above.

outprefix: output filename prefix

nodecount: maximum number of bins. 1, \ldots, nodecount.

nooverlap: whether predicted domains may overlap with each other.

prepromodel: marker preprocessing






