
from time import sleep
from sklearn.calibration import CalibratedClassifierCV as cc, calibration_curve
from Bio import SeqIO
from pydpi.pypro import PyPro
import sys
import numpy as np
import warnings
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
from sklearn import svm, datasets, metrics
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, precision_recall_curve, precision_recall_fscore_support
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score, recall_score, roc_auc_score, auc, matthews_corrcoef, classification_report
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.decomposition import PCA, TruncatedSVD as svd
from scipy import interp
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, SelectFromModel
#from classifier.classical_classifiers import RFClassifier, SVM
from make_representations.sequencelist_representation import SequenceKmerRep, SequenceKmerEmbRep
from sklearn.metrics.scorer import make_scorer
import argparse
import pickle
import errno




protein = PyPro()


def readAAP(file):  # read AAP features from the AAP textfile
    try:
        aapdic = {}
        aapdata = open(file, 'r')
        for l in aapdata.readlines():
            aapdic[l.split()[0]] = float(l.split()[1])
        aapdata.close()
        return aapdic
    except:
        print("Error in reading AAP feature file. Please make sure that the AAP file is correctly formatted")
        sys.exit()


def readAAT(file):  # read AAT features from the AAT textfile
    try:
        aatdic = {}
        aatdata = open(file, 'r')
        for l in aatdata.readlines():
            aatdic[l.split()[0][0:3]] = float(l.split()[1])
        aatdata.close()
        return aatdic
    except:
        print("Error in reading AAT feature file. Please make sure that the AAT file is correctly formatted")
        sys.exit()


def aap(pep, aapdic, avg):  # return AAP features for the peptides
    feature = []
    for a in pep:
        # print(a)
        if int(avg) == 0:
            score = []
            count = 0
            for i in range(0, len(a) - 1):
                try:
                    score.append(round(float(aapdic[a[i:i + 2]]), 4))
                    # score += float(aapdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    # print(a[i:i + 3])
                    score.append(float(-1))
                    # score += -1
                    count += 1
                    continue
            # averagescore = score / count
            feature.append(score)
        if int(avg) == 1:
            score = 0
            count = 0
            for i in range(0, len(a) - 1):
                try:
                    score += float(aapdic[a[i:i + 2]])
                    count += 1
                except KeyError:
                    score += -1
                    count += 1
                    continue
            if count != 0:
                averagescore = score / count
            else:
                averagescore = 0
            feature.append(round(float(averagescore), 4))
    return feature


def aat(pep, aatdic, avg):  # return AAT features for the peptides
    feature = []
    for a in pep:
        if int(avg) == 0:
            # print(a)
            score = []
            count = 0
            for i in range(0, len(a) - 2):
                try:
                    score.append(round(float(aatdic[a[i:i + 3]]), 4))
                    # score += float(aapdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    # print(a[i:i + 3])
                    score.append(float(-1))
                    # score += -1
                    count += 1
                    continue
            # averagescore = score / count
            feature.append(score)
        if int(avg) == 1:
            score = 0
            count = 0
            for i in range(0, len(a) - 2):
                try:
                    score += float(aatdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    score += -1
                    count += 1
                    continue
            # print(a, score)
            if count != 0:
                averagescore = score / count
            else:
                averagescore = 0
            feature.append(round(float(averagescore), 4))
    return feature


def CTD(pep):  # Chain-Transition-Ditribution feature
    feature = []
    name = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        ctd = protein.GetCTD()
        feature.append(list(ctd.values()))
        name = list(ctd.keys())
    return feature, name


def AAC(pep):  # Single Amino Acid Composition feature
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        aac = protein.GetAAComp()
        feature.append(list(aac.values()))
        name = list(aac.keys())
    return feature, name


def DPC(pep):  # Dipeptide Composition feature
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        dpc = protein.GetDPComp()
        feature.append(list(dpc.values()))
        name = list(dpc.keys())
    return feature, name


def PAAC(pep):
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        #paac=protein.GetMoranAuto()
        paac = protein.GetPAAC(lamda=4)
        feature.append(list(paac.values()))
        name = list(paac.keys())
    return feature, name


def kmer(pep, k):  # Calculate k-mer feature
    feature = SequenceKmerRep(pep, 'protein', k,norm='l1')
    return feature


def protvec(pep, k, file):  # Calculate ProtVec representation
    feature = SequenceKmerEmbRep(file, pep, 'protein', k)
    return feature


def QSO(pep):
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        #paac=protein.GetMoranAuto()
        qso = protein.GetQSO(maxlag=5)
        feature.append(list(qso.values()))
        name = list(qso.keys())
    return feature, name



def read_peptides(pos_file, neg_file): 
    """ return the peptides from input peptide list file """ 

    print(">> Start sequence to read the files contains the positive and negative peptides")
    
    # Try open the files
    try:
        pos_data = open(pos_file,'r')
        neg_data = open(neg_file,'r')
    except ValueError:
        sys.exit()

    pos = []
    for l in pos_data.readlines():
        if l[0] == '#': # Ignoring the comments on file
            continue
        else:
            pos.append(l.strip()) # Append to positive list the peptide and remove all spaces
    
    neg = []
    for l in neg_data.readlines():
        if l[0] == '#': # Ignoring the comments on file
            continue
        else:
            neg.append(l.strip()) # Append to negative list the peptide and remove all spaces

    print(">> Finish sequence to read the files contains the positive and negative peptides")

    return pos, neg


def combinefeature(pep, featurelist, dataset):

    print(">> Start the extraction of the features")

    a=np.empty([len(pep), 1])
    fname=[]
    scaling = StandardScaler()
    #pca = svd(n_components=300)
    pca = PCA(0.99)
    vocab_name = []
    #pca = PCA(n_components=10)
    #print(a)
    
    if 'aap' in featurelist:
        print(f">>> Extract AAP to {len(pep)} peptides!")
        aapdic = readAAP("aap/aap-viral.normal")
        f_aap = np.array([aap(pep, aapdic, 1)]).T
        a = np.column_stack((a,f_aap))
        #a = scaling.fit_transform(a)
        fname.append('AAP')
        #print(f_aap)
    
    if 'aat' in featurelist:
        print(f">>> Extract AAT to {len(pep)} peptides!")
        aatdic = readAAT("aat/aat-viral.normal")
        f_aat = np.array([aat(pep, aatdic, 1)]).T
        a = np.column_stack((a, f_aat))
        #a = scaling.fit_transform(a)
        fname.append('AAT')
        #print(f_aat)
    
    if 'dpc' in featurelist:
        print(f">>> Extract DPC to {len(pep)} peptides!")
        f_dpc, name = DPC(pep)
        # f_dpc = np.average(f_dpc, axis =1)
        a = np.column_stack((a, np.array(f_dpc)))
        fname = fname + name
    
    if 'aac' in featurelist:
        print(f">>> Extract AAC to {len(pep)} peptides!")
        f_aac, name = AAC(pep)
        a = np.column_stack((a, np.array(f_aac)))
        fname = fname + name
    
    if 'paac' in featurelist:
        print(f">>> Extract PAAC to {len(pep)} peptides!")
        f_paac, name = PAAC(pep)
        #f_paac = pca.fit_transform(f_paac)
        a = np.column_stack((a, np.array(f_paac)))
        fname = fname + name
    
    if 'kmer' in featurelist:
        print(f">>> Extract Kmer to {len(pep)} peptides!")
        kmers = kmer(pep, 2)
        #f_kmer = np.array(kmers.X.toarray())
        f_kmer = np.array(kmers.X.toarray())
        vocab_name = kmers.vocab

        a = np.column_stack((a, f_kmer))
        fname = fname + ['kmer']*len(f_kmer)
    
    if 'qso' in featurelist:
        print(f">>> Extract QSO to {len(pep)} peptides!")
        f_qso, name = QSO(pep)
        #f_pa = pca.fit_transform(f_paac)
        a = np.column_stack((a, np.array(f_qso)))
        fname = fname + name

    if 'ctd' in featurelist:
        print(f">>> Extract CTD to {len(pep)} peptides!")
        f_ctd, name = CTD(pep)
        a = np.column_stack((a, np.array(f_ctd)))
        fname = fname + name

    if 'protvec' in featurelist:
        print(f">>> Extract Protvec to {len(pep)} peptides!")
        f_protvec = np.array(protvec(pep, 4, './protvec/sp_sequences_4mers_vec.bin').embeddingX)
        #f_protvec = pickle.load(open("features_protvec.pickle", 'rb'))
        #f_protvec = np.average(f_protvec, axis =1)
        a = np.column_stack((a, f_protvec))
        fname = fname + ['protvec']*len(f_protvec)

    print(">> Finish the extraction of the features")

    return a[:,1:], fname, vocab_name


def run_training(pos, neg, dataset, savename):
    """ Preparation to training the model with dataset name, positive data and negative data send by parameters"""
    
    print(">> Start the preparation to run the training")
    
    pep_combined = pos + neg
    
    print(f">>> Total peptides on the dataset: {len(pep_combined)}")
    
    pickle_info={}
    # featurelist = ['aac','aap','aat','protvec', 'dpc', 'kmer']
    featurelist = ['aac']
    pickle_info['featurelist'] = featurelist
    features, fname, vocab = combinefeature(pep_combined, featurelist, dataset)
    
    print(f">>> Total features extract per peptide: {len(features[0])}")
    
    pickle_info['feat_name'] = fname
    pickle_info['vocab'] = vocab
    #pickle.dump(features, open("features_latest.pickle", "wb"))
    #print(features)
    target = [1] * len(pos) + [0] * len(neg)
    #print(pep_combined)
    train(pep_combined, features, target, pickle_info, dataset, savename)


def precision_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate prec for neg class
    '''
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return p

def recall_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate recall for neg class
    '''
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return r

def f1_0(y_true, y_pred, labels=None, average='binary', sample_weight=None):
    '''
    :param y_true:
    :param y_pred:
    :param labels:
    :param average:
    :param sample_weight:
    :return: calculate f1 for neg class
    '''
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=1,
                                                 labels=labels,
                                                 pos_label=0,
                                                 average=average,
                                                 warn_for=('f-score',),
                                                 sample_weight=sample_weight)
    return f


def gridsearch(x, y, cv):

    print(">> Start train grid search to model")

    scoring = { 'auc_score': 'roc_auc',
                'accuracy': 'accuracy',
                'scores_p_1': 'precision',
                'scores_r_1': 'recall',
                'scores_f_1_1': 'f1',
                'scores_p_0': make_scorer(precision_0),
                'scores_r_0': make_scorer(recall_0),
                'scores_f_1_0': make_scorer(f1_0),
                'mcc': make_scorer(matthews_corrcoef),
                'precision_micro': 'precision_micro',
                'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro',
                'recall_micro': 'recall_micro', 'f1_macro': 'f1_macro', 'f1_micro': 'f1_micro'}
                
    grid_search = GridSearchCV(SVC(kernel='rbf', probability=True),
                               param_grid={'C': [1000, 500, 250, 100, 50, 25, 1, 0.1,
                                                 0.01, 0.001, 0.0001],
                                           'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]},
                               scoring=scoring, cv=cv, n_jobs=40, refit='auc_score',verbose=2)

    # grid_search = GridSearchCV(SVC(kernel='rbf', probability=True),
    #                            param_grid={'C': [1000],
    #                                        'gamma': [100]},
    #                            scoring=scoring, cv=cv, n_jobs=40, refit='auc_score',verbose=2)

    grid_search.fit(x, y)

    print(">> Finish train grid search to model")

    return grid_search


def gridsearch_linear(x, y, cv):
    # 1000, 500, 200, 100, 50,20, 10, 2, 1, 0.2, 0.5,0.01, 0.02, 0.05, 0.001
    '''
    param_grid={'C': [1000, 500, 200, 100, 50,
                                                 20, 10, 2, 1, 0.2, 0.5,
                                                 0.01, 0.02, 0.05, 0.001],
                                           'gamma': [1000, 500, 200, 100,
                                                     50, 20, 10, 5, 2, 1,
                                                     0.2, 0.5, 0.01, 0.02,
                                                     0.05, 0.001, 0.0001]},'''
    scoring = { 'auc_score': 'roc_auc',
                'accuracy': 'accuracy',
                'scores_p_1': 'precision',
                'scores_r_1': 'recall',
                'scores_f_1_1': 'f1',
                'scores_p_0': make_scorer(precision_0),
                'scores_r_0': make_scorer(recall_0),
                'scores_f_1_0': make_scorer(f1_0),
                'mcc': make_scorer(matthews_corrcoef),
                'precision_micro': 'precision_micro',
                'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro',
                'recall_micro': 'recall_micro', 'f1_macro': 'f1_macro', 'f1_micro': 'f1_micro'}
    grid_search = GridSearchCV(LinearSVC(max_iter=1000),
                               param_grid={ 'penalty' : ['l2'],
                                            'C': [1000, 500, 200, 100, 50,
                                                 20, 10, 2, 1, 0.2, 0.5,
                                                 0.01, 0.02, 0.05, 0.001]},
                               scoring=scoring, cv=cv, n_jobs=40, refit='auc_score',verbose=2)
    '''grid_search = GridSearchCV(LinearSVC(max_iter=1000),
                               param_grid={ 'penalty' : ['l2'],
                                            'C': [1000, 500, 200, 100, 50,
                                                 20, 10, 2, 1, 0.2, 0.5,
                                                 0.01, 0.02, 0.05, 0.001]},
                               scoring={'accuracy','roc_auc'}, cv=cv, n_jobs=-1, refit='accuracy')
    grid_search = GridSearchCV(SVC(kernel='rbf', cache_size=2000, probability=True),
                               param_grid={'C': [10000, 5000, 1],
                                           'gamma': ['scale']},
                               scoring={'accuracy','roc_auc'}, cv=cv, n_jobs=-1, refit='accuracy')'''
    grid_search.fit(x, y)
    return grid_search


def train(peptides, features, target, pickle_info, dataset, savename):
    scaling = StandardScaler()
    scaling.fit(features)
    print(max(features[:,0]))
    x = scaling.transform(features)
    #print(max(x[:,1]))
    y = np.array(target)
    
    cv = StratifiedKFold(n_splits=5)
    model = gridsearch(x, y, cv)
    
    aapdic = readAAP("aap/aap-viral.normal")
    aatdic = readAAT("aat/aat-viral.normal")
    
    pickle_info ['aap'] = aapdic
    pickle_info ['aat'] = aatdic
    pickle_info ['scaling'] = scaling
    pickle_info ['model'] = model
    pickle_info ['training_features'] = features
    pickle_info ['training_targets'] = y

    filename = "model_train/"+dataset+"/svm-"+dataset+savename+".pickle"
    
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    pickle.dump(pickle_info, open("model_train/"+dataset+"/svm-"+dataset+savename+".pickle", "wb"))
    print("Best parameters: ", model.best_params_)
    print("Best accuracy: :", model.best_score_)
    results = model.cv_results_
    bi = model.best_index_
    print("roc_auc:",results['mean_test_auc_score'][bi],
          "accuracy:",results['mean_test_accuracy'][bi],
          "precision +:",results['mean_test_scores_p_1'][bi],
          "recall +:",results['mean_test_scores_r_1'][bi],
          "f1 +:",results['mean_test_scores_f_1_1'][bi],
          "precision -:",results['mean_test_scores_p_0'][bi],
          "recall -:",results['mean_test_scores_r_0'][bi],
          "f1 -:",results['mean_test_scores_f_1_0'][bi],
          "precision_micro:",results['mean_test_precision_micro'][bi],
          "f1 -:",results['mean_test_precision_macro'][bi],
          "mcc -:",results['mean_test_mcc'][bi])

    '''MRF = RFClassifier(x, y)
    MRF.tune_and_eval("4mer_rf")'''


def readmodel(mlfile):
    try:
        return pickle.load(open(mlfile, 'rb'))
    except:
        print("Error in reading model file")
        sys.exit()


def predict(model, features):
    try:
        return model.predict_proba(features)
    except:
        print("Error in predicting epitopes.")
        sys.exit()


def scoremodel(file, mlfile):
    sequence = readseq(file)
    pep = peptides(sequence)
    features = combinefeature(pep)
    # print(len(features[0]))
    model = readmodel(mlfile)
    return pep, predict(model, features)

def get_parameters_command_line():
    """ Get all parameters from command line and return"""

    print(">> Start sequence to get all parameters from command line")

    # Verify if the positive flag to positive file of sequences is present
    if "-p" in sys.argv:
        positive_flag = sys.argv.index("-p")
        positive_file = sys.argv[positive_flag + 1]
    else:
        print("ERROR: The positive file is not found.")
        sys.exit()

    # Verify if the negative flag to negative file of sequences is present
    if "-n" in sys.argv:
        negative_flag = sys.argv.index("-n")
        negative_file = sys.argv[negative_flag + 1]
    else:
        print("ERROR: The negative file is not found.")
        sys.exit()

    # Verify if the dataset flag to dataset name is present
    if "-d" in sys.argv:
        dataset_flag = sys.argv.index("-d")
        dataset_name = sys.argv[dataset_flag + 1]
    else:
        print("ERROR: The dataset name is not found.")
        sys.exit()
    
    # Verify if the savename flag to savename is present
    if "-s" in sys.argv:
        savename_flag = sys.argv.index("-s")
        savename = sys.argv[savename_flag + 1]
    else:
        print("ERROR: The savename is not found.")
        sys.exit()

    print(">> Finish the sequence to get all parameters from command line")

    return positive_file, negative_file, dataset_name, savename


if __name__ == "__main__":

    print(">> Start algorithm")

    pos_file, neg_file, dataset, savename = get_parameters_command_line()
    pos_data, neg_data = read_peptides(pos_file, neg_file)

    run_training(pos_data, neg_data, dataset, savename)
