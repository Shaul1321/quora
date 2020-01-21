
from typing import Dict
import numpy as np
import scipy
import siamese

from typing import List
from tqdm import tqdm
import random
import warnings
import sklearn
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import copy

def get_nullspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix
    """
    nullspace_basis = scipy.linalg.null_space(W)  # orthogonal basis

    nullspace_basis = nullspace_basis * np.sign(nullspace_basis[0][0])  # handle sign ambiguity
    projection_matrix = nullspace_basis.dot(nullspace_basis.T)

    return projection_matrix
    
    
def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix
    """

    w_basis = scipy.linalg.orth(W.T) # orthogonal basis
    w_basis * np.sign(w_basis[0][0]) # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace
    
    return P_W



def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):

    rowspace_projections = []
    I = np.eye(input_dim)
    
    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    Q = np.sum(rowspace_projections, axis = 0)
    P = I - get_rowspace_projection(Q)
    
    return P


def get_debiasing_projection(num_classifiers: int, input_dim: int,
                             is_autoregressive: bool, X_train: np.ndarray, X_dev: np.ndarray, dropout_rate = 0, device = "cpu") -> np.ndarray:
    """
    :param classifier_class: the sklearn classifier class (SVM/Perceptron etc.)
    :param cls_params: a dictionary, containing the params for the sklearn classifier
    :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
    :param input_dim: size of input vectors
    :param is_autoregressive: whether to train the ith classiifer on the data projected to the nullsapces of w1,...,wi-1
    :param min_accuracy: above this threshold, ignore the learned classifier
    :param X_train: ndarray, training vectors
    :param Y_train: ndarray, training labels (protected attributes)
    :param X_dev: ndarray, eval vectors
    :param Y_dev: ndarray, eval labels (protected attributes)
    :param by_class: if true, at each iteration sample one main-task label, and extract the protected attribute only from vectors from this class
    :param T_train_main: ndarray, main-task train labels
    :param Y_dev_main: ndarray, main-task eval labels
    :param dropout_rate: float, default: 0 (note: not recommended to be used with autoregressive=True)
    :return: P, the debiasing projection; rowspace_projections, the list of all rowspace projection; Ws, the list of all calssifiers.
    """
    if dropout_rate > 0 and is_autoregressive:
        warnings.warn("Dropout is not recommended with autoregressive training, as it violates the propety w_i.dot(w_(i+1)) = 0 that is necessary for a mathematically valid projection.")
    
    I = np.eye(input_dim)
 
    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []
    
    pbar = tqdm(range(num_classifiers))
    for i in pbar:
        print("======================================")
        clf = siamese.Siamese(X_train_cp, X_dev_cp, 100, batch_size = 1000, dropout_rate = dropout_rate, device = device).to(device)
        
        acc = clf.train_network(20)
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))

        W = clf.get_weights()

        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W) # projection to W's rowspace
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:
            
            """
            to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far (instaed of doing X = P_iX,
            which is problematic when w_i is not exactly orthogonal to w_i-1,...,w1, due to e.g inexact argmin calculation).
            """
            # use the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf: 
            # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
            
            Q = np.sum(rowspace_projections, axis = 0)
            P = I - get_rowspace_projection(Q)
            
            # project  
                      
            X_train_cp = (P.dot(X_train.T)).T
            X_dev_cp = (P.dot(X_dev.T)).T

    """
    calculae final projection matrix P=PnPn-1....P2P1
    since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
    by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability,
    i.e., we explicitly project to intersection of all nullspaces (not very critical)
    """
    
    Q = np.sum(rowspace_projections, axis = 0)
    P = I - get_rowspace_projection(Q)

    return P, rowspace_projections, Ws


if __name__ == '__main__':

        #net = siamese.Siamese(np.random.rand(1000,1024), np.random.rand(1000,1024), 32, 100, dropout_rate = 0.5)
        x = np.random.rand(1000,1024) - 0.5
        y = np.random.rand(1000,1024) - 0.5
        s = ["a"] * len(x)
        ids = [1] * len(x)
        data = list(zip(x,y,s,s,ids))
        
        get_debiasing_projection(1, 1024, is_autoregressive = False, X_train = data, X_dev = data, dropout_rate = 0.5)
