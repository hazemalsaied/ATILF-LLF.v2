You can find in this folder the script used to generate the results of baseline system. 

You can run one of three possible experiments: 
baseline on test data set : the function identify of the script.
baseline 5-folds cross validation on train data set: the function identifyForCV of the script, with onTrainDataSet parameter activated.
baseline 5-folds cross validation on the whole corpus(train + test): the function identifyForCV of the script.

The result and golden mwe files would be stored in the result folder. 

we provide also two sh scripts to use the mwe sharedtask 2017 evaluation scripts, which exists in Scripts folder.
 
 There is also a function, fromTxt2CSV, in the baseline.py script which could convert the textual output of shredtask scripts into .csv file.
