import os
import optparse
parser = optparse.OptionParser()
parser.add_option('-i', '--inputFiles',     help="Input files")
parser.add_option('-n', '--names',          help="Legend Names")
parser.add_option('-o', '--outdir',     default='', type=str, help='outputDirectory')
o, a = parser.parse_args()

outputDir = o.outdir
if not os.path.isdir(outputDir):
    print("Making output dir",outputDir)
    os.mkdir(outputDir)


print(o.inputFiles)

#"ZZ4b/nTupleAnalysis/pytorchModels/3bMix4bv1FvT_ResNet+multijetAttention_9_9_9_np2070_lr0.008_epochs40_stdscale.log"]
inputFileNames = o.inputFiles.split(",")
labels         = o.names.split(",")

if not len(inputFileNames) == len(labels):
    print("Number of input files and name have to be the same!")
    print("you gave ",len(inputFileNames),"vs", len(labels))
    print("Exiting...")
#for i in o.inputFiles.split.:
#    inputFiles.append(i)

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def readLogFile(infileName,label):
    print("Processing",infileName)
    infile = open(infileName,"r")
    foundValidStart = False

    for line in infile:
        words = line.split()
        
        if not len(words): continue
        if words[0] == ">>":
            if words[1] == "Epoch":   
                foundValidStart = True
                data={}
                data["file"] = infileName.split("/")[-1]
                data["label"] = label
                data["epochs"] = []
                data["val_loss"] = []
                data["val_norm"] = []
                data["val_AUC"] = []
                data["overTrain"] = []
                data["train_loss"] = []
                data["train_norm"] = []
                data["train_AUC"] = []
                epoch = 0
                continue

            epoch = int(words[1].split("/")[0])
            data["epochs"].append(epoch)
            data["val_loss"].append(float(words[5]))
            data["val_norm"].append(float(words[7]))
            data["val_AUC"].append(float(words[9]))
            if epoch == 0:
                data["overTrain"].append(0)
            else:
                data["overTrain"].append(float(words[12].replace("%","")))

        elif foundValidStart and words[0] == "Training":
            data["train_loss"].append(float(words[2]))
            data["train_norm"].append(float(words[4]))
            data["train_AUC"].append(float(words[6]))


    return data
            

def makePlot(name,inputData,xKey,yKey,estart,yTitle,logy=False,xTitle="Epoch",yMax=None,yMin=None):
    plt.figure(figsize=(10,7))
    if logy:
        plt.yscale('log')


    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    if yMax:
        plt.ylim(yMin,yMax)
    for ds in range(len(inputData)):
        if not ds:
            plt.plot(inputData[0][xKey][estart:], inputData[0][yKey][estart:],marker="o",linestyle="-",color="r",label=inputData[0]["label"])
        else:
            plt.plot(inputData[ds][xKey][estart:], inputData[ds][yKey][estart:],marker="o",linestyle=":",label=inputData[ds]["label"])
    plt.legend(loc="best")
    plt.savefig(outputDir+"/"+name+".pdf")


def plotData(inputData,estart=0):
    #makePlot("Val_Loss",  inputData,"epochs","val_loss",  estart,"Validation Loss",yMax=0.1675, yMin=0.16)
    makePlot("Val_Loss",  inputData,"epochs","val_loss",  estart,"Validation Loss")
    makePlot("Val_Loss_l",  inputData,"epochs","val_loss",  estart,"Validation Loss",logy=True)
    makePlot("Val_Norm",  inputData,"epochs","val_norm",  estart,"Validation Norm")
    makePlot("Val_AUC",   inputData,"epochs","val_AUC",   estart,"Validation AUC")

    #makePlot("Train_Loss",  inputData,"epochs","train_loss",estart,"Training Loss",yMax=0.1675, yMin=0.16)
    makePlot("Train_Loss",  inputData,"epochs","train_loss",estart,"Training Loss")
    makePlot("Train_Loss_l",inputData,"epochs","train_loss",estart,"Training Loss",logy=True)
    makePlot("Train_Norm",  inputData,"epochs","train_norm",estart,"Training Norm")
    makePlot("Train_AUC",   inputData,"epochs","train_AUC", estart,"Training AUC")

    makePlot("overTrain", inputData,"epochs","overTrain", estart,"Over Training Metric")



inputData = []

for itr, inName in enumerate(inputFileNames):
    inputData.append(readLogFile(inName,labels[itr]))




plotData(inputData,estart=1)

#print(inputData)
#plt.ylim(ratioRange)
#plt.xlim([bins[0],bins[-1]])
#plt.plot([bins[0], bins[-1]], [1, 1], color='k', alpha=0.5, linestyle='--', linewidth=1)
