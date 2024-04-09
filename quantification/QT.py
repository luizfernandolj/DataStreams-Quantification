import os
import pandas as pd
from utils.pandas_to_arff import pandas_arff

def QT(test, dts_name,folder):
    """Quantification Trees

    It quantifies events based on Decision trees adopted for quantification purpose, applying QT, according to Milli et al.(2013).
    
    Parameters
    ----------
    test : dataframe
        A DataFrame of the test data.
      
    Returns
    -------
    array
        the class distribution of the test. 
    """

    test = pd.DataFrame(test)
    test['class'] = test['class'].astype('str')
    pandas_arff(df=test, filename= folder + "/test_data_%s" % dts_name + ".arff", wekaname=dts_name)
    # The next lines apply the QT model (built in the previous step) over the test (X_test) that was converted into ARFF file in the previous line
    # NOTE: the result of the quantification will be writting into an output file. In this example the output file is re.txt. We need to grab the result from this file.
    command = "java  -Xmx5G -cp quantify.jar:weka.jar:. weka.classifiers.trees.RandomForest -l "
    #command+= "./models_train_test/"+dts_name+"/classifier -T ./models_train_test/"+dts_name+"/test.arff"
    command+= folder +"/classifier" + " -T " + folder + "/test_data_%s" % dts_name + ".arff"
    
    #the next line save the name of the result file. We need it to grab the results in the next step
    #result_file = "./models_train_test/"+dts_name+"/re.txt"
    result_file = folder + "/re.txt"
    command+= " > "+result_file
    print(command)
    os.system(command)
    # The next lines open the result file
    f=open(result_file)
    lines=f.readlines()
    print(lines)
    f.close()
    cc=float(lines[13].split(":")[1].split(" ")[1].split("%")[0])/100
    
    return cc