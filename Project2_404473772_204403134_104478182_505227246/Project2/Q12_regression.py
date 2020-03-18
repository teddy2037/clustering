# This is a separate regression file meant only for question 12

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix

from sklearn.metrics import homogeneity_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import completeness_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score

import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn import preprocessing

from sklearn.decomposition import NMF

import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def print_metrics_and_return(Bin_Target_Test, pred):
    m1 = (homogeneity_score(Bin_Target_Test, pred))
    m2 = (v_measure_score(Bin_Target_Test, pred))
    m3 = (completeness_score(Bin_Target_Test, pred))
    m4 = (adjusted_rand_score(Bin_Target_Test, pred))
    m5 = (adjusted_mutual_info_score(Bin_Target_Test, pred))
    return list([m1, m2, m3, m4, m5])



def scalingfeatures(X):
    X_scaled = preprocessing.scale(X, axis=0, with_mean=True, with_std=True)
    return X_scaled

def logscale(X,c):
    X_sign  = np.sign(X)
    X_abs = np.absolute(X)
    return X_sign*(np.log(X_abs+c)-np.log(c))

  
  
def scalingmetric(X, Target, c, best_r, scale = 0, log = 0, order = 0, flag = False, n_cluster_flex = 2):
    if order == 0:
        if scale == 0:
            X_1 = scalingfeatures(X)
        else:
            X_1 = X

        if log == 0:
            X_2 = logscale(X_1, c)
        else:
            X_2 = X_1

    else:
        X_1 = logscale(X,c)
        X_2 = scalingfeatures(X_1)

    kmeans = KMeans(n_clusters = n_cluster_flex, n_init=30, max_iter=1000, random_state=None).fit(
        X_2)
    pred = kmeans.predict(X_2)
    # print pred
    # print Bin_Target_Test

    if flag == True:
        print(print_metrics_and_return(Target, pred))
        plotKmeans(X_2, pred, Target, best_r)
    
    if flag == False:
        # print print_metrics_and_return(Bin_Target_Test, pred)
        return print_metrics_and_return(Target, pred)
    else:
        return 0,0,0,0,0





def init_file(flag = False):

    dataset_all_cat = fetch_20newsgroups(subset = 'all', shuffle = True, random_state = 0)

    Target = dataset_all_cat.target
    count_vect_all = CountVectorizer(min_df=3, stop_words='english')

    X_counts_all = count_vect_all.fit_transform(dataset_all_cat.data)

    tfidf_transformer_all = TfidfTransformer()
    X_tfidf_all = tfidf_transformer_all.fit_transform(X_counts_all)

    if flag == True:
        print("X_tfidf size: ", X_tfidf_all.shape)

    return X_tfidf_all, Target

"""## Question 12: Pipeline the Whole Damn Thing"""



def best_metrics_r_svd(X_svd, Target, r):
    v_meas = -10.0
    op_best = []
    
    for order in range(0,2):
        for scale in range(0,2):
            for log in range(0,2):
                if order == 1 and (scale != 1 or log != 1):
                    continue
                op = scalingmetric(X_svd, Target, 0.01, r, scale, log, order, False, 20)
                if v_meas < op[1]:
                    op.extend([scale, log, order])
                    op_best = op
                    v_meas = op[1]
                op = []
    return op_best


def best_metrics_r_nmf(X_nmf, Target, r):
    v_meas = -10.0
    op_best = []

    for order in range(0,2):
        for scale in range(0,2):
            for log in range(0,2):
                if order == 1 and (scale != 1 or log != 1):
                    continue
                op = scalingmetric(X_nmf, Target, 0.01, r, scale, log, order, False, 20)
                if v_meas < op[1]:
                    op.extend([scale, log, order])
                    op_best = op
                    v_meas = op[1]
                op = []
    return op_best


def collect_metrics_svd_nmf_q12():

    # initializing parameters
    X_tfidf, Target = init_file()

    opt_metrics_vmeas = -10.0
    for i in [1,2,3,5,10,20,50,100,300,500,700]:
        print "Begin " + str(i) + "th regression..."
        # dimensionality reduction for all r
        t_svd12 = TruncatedSVD(n_components=i, n_iter=7, random_state=42)
        t_nmf12 = NMF(n_components=i, init='random', random_state=42)

        X_svd12 = t_svd12.fit_transform(X_tfidf)
        X_nmf12 = t_nmf12.fit_transform(X_tfidf)


        # we will use v_measure score because it captures both the effects of
        # homogeneity and completeness. Further it is invariant to permutations
        # in actual cluster labels and symmetric in testv. training or train v. test.

        # Finding the optimal svm, r metric for all scenarios of log-scale
        print "Researching best " + str(i) + "_SVM metrics: (Yeah!)"
        opt_metrics_svd = best_metrics_r_svd(X_svd12, Target, i)

        # Finding the optimal nmf, r metric for all scenarios of log-scale
        print "Researching best " + str(i) + "_NMF metrics: (Nah!)"
        opt_metrics_nmf = best_metrics_r_nmf(X_nmf12, Target, i)

        if opt_metrics_vmeas < opt_metrics_svd[1]:
            opt_metrics_vmeas = opt_metrics_svd[1]
            print "The new best parameters with " + str(i) + ", SVD"
            print opt_metrics_svd

        if opt_metrics_vmeas < opt_metrics_nmf[1]:
            opt_metrics_vmeas = opt_metrics_nmf[1]
            print "The new best parameters with " + str(i) + ", NMF"
            print opt_metrics_nmf

if __name__ == "__main__":
    collect_metrics_svd_nmf_q12()

