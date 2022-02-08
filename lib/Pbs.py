import os
import sys

class Pbs():
    """ methods related to pbs
    """
    def __init__(self):
        return

    @staticmethod
    def submitPbs(code2run,pbsfolder,pbsfilename,queue):
        pbsfilepath = "{0}/{1}.pbs".format(pbsfolder,pbsfilename)
        errorpath = "{0}/{1}.err".format(pbsfolder,pbsfilename)
        outpath = "{0}/{1}.out".format(pbsfolder,pbsfilename)
        file = open(pbsfilepath,"w")
        file.write("#!/bin/sh\n")
        file.write("#PBS -l nodes=1:ppn=2\n")
        file.write("#PBS -l walltime=16:00:00\n")
        file.write("#PBS -q {0}\n".format(queue))
        file.write("#PBS -r n\n")
        file.write("#PBS -V\n")
        file.write("#PBS -o {0}\n".format(outpath))
        file.write("#PBS -e {0}\n".format(errorpath))
        file.write("cd $PBS_O_WORKDIR\n")
        file.write(code2run+"\n")
        file.close()
        code = "qsub {0}".format(pbsfilepath)
        os.system(code)
