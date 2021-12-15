import torch
import torch.nn as nn
import os
import numpy as np

import torch.nn.functional as F
from loss import SupConLoss
from sklearn.metrics import roc_curve, auc
from torch.cuda.amp import autocast

from sklearn.metrics import roc_curve
# from torch.cuda.amp import autocast

def cosine_similarity(a, b, eps=1e-8):
    # a: probe: (Num_probes, embedding)
    # b: gallery: (Num_Subjects (90 for FVG), embedding)

    # Compute pairwisecosine similarities of every probe element with gallery element, return a matrix of size (Num_probes, num_subjects), containing similarities of every probe with every gallery sequence

    a = torch.from_numpy(a).cuda()
    b = torch.from_numpy(b).cuda()

    assert a.shape[1] == b.shape[1]

    # Logic: cosine_similarity(u,v) = dot(u,v) / (norm(u) * norm(v))
    #                               = dot(u / norm(u), v / norm(v))

    # added eps for numerical stability
    
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

    # sim_mt is of shape(num_probes, num_gallery_sequences (90 for FVG))
    return sim_mt.cpu().numpy()

def find_idx(fpr, tpr, desired_fpr=[0.01, 0.05, 0.1], ifround=False):
    # Takes the list fpr, tpr. Returns TPR at the desired FPR (1%, 5%, 10%)
    outptut = []
    for i in desired_fpr:
        item = fpr[fpr < i + 0.005].max()
        idx = np.where(fpr == item)
        val = tpr[idx][-1]
        if ifround:
            val = round(val, 2)
        outptut.append(val)
    return outptut

def get_probe_mask(protocol, condition_list, session_list):
    # Returns the probe mask according to the desired protocol. 
    # probe_mask filters which sequences to consider as probe. Eg. for CB, select 10,11,12 from condition_list (those with 1 in session_list)


    ## Probe sequences: 
    ## WS: Session 1: 4-9
    #      Session 2: 4-6
    # 
    # CB (Bag): S1: 10-12
    # 
    # CL (Clothing): S2: 7-9
    # 
    # CBG (Background) : S2: 10-12
    
    # All: 1, 3-12 in both sessions

    assert protocol in ['ws', 'all', 'cb', 'cl', 'cbg'], "Unknown testing protocol"

    if protocol == 'ws':
        probe_sequences_s1 = [4,5,6,7,8,9]
        probe_sequences_s2 = [4,5,6]
        probe_mask = (np.isin(condition_list, probe_sequences_s1) & (np.isin(session_list, [1]))) | (np.isin(condition_list, probe_sequences_s2) & (np.isin(session_list, [2])))

    elif protocol == 'all':
        probe_sequences = [1,3,4,5,6,7,8,9,10,11,12]
        probe_mask = np.isin(condition_list, probe_sequences)

    else:
        if protocol == 'cb':
            probe_sequences = [10,11,12]
            session = 1
        elif protocol == 'cl':
            probe_sequences = [7,8,9]
            session = 2
        elif protocol == 'cbg':
            probe_sequences = [10,11,12]
            session = 2
        
        probe_mask = np.isin(condition_list, probe_sequences) & (np.isin(session_list, [session]))

    return probe_mask
        

def get_scores(protocol, features, labels, condition_list, session_list):
    # HOW TO CALCULATE SCORES
    # Compute (cosine) similarity of every probe with every gallery. Shape: (# of probes, # of gallery)
    # Convert Ground truth labels (subject ids) of probes to one hot encoding. Shape: (# of probes, # of gallery)
    # Flatten both similarity matrix and label metrix
    # sklearn.roc_curve() to get true positive rates and false positive rates
    # find TPR at 1% fpr, 5% fpr to get score.

    
    gallery_mask = (np.isin(condition_list, [2]))
    gallery_features = features[gallery_mask, :]
    gallery_labels = labels[gallery_mask]
    gallery_length = len(gallery_labels)


    #print("Getting masks..")
    probe_mask = get_probe_mask(protocol, condition_list, session_list)
    # probe_mask filters which sequences to consider as probe. Eg. for CB, select 10,11,12 from condition_list (those with 1 in session_list)
    #print("masks calculated")
   
    probe_features = features[probe_mask, :]
    probe_labels = labels[probe_mask]
    probe_length = len(probe_labels)

    similarities = cosine_similarity(probe_features, gallery_features)
    # similarities[i, j] is the distance between ith probe and jth gallery element
    # similarities is still a numpy array
    
    ############### Compute accuracy #########################
    sorted_sim_indices = np.argsort(similarities, axis=1) #Sort along gallery dimension
    # The last index along gallery dimension gives the id predictions
    last_index = sorted_sim_indices.shape[1] - 1
    id_predictions = sorted_sim_indices[:,last_index]

    accuracy = np.sum(gallery_labels[id_predictions] == probe_labels)/float(probe_length)

    #########################################################

    probe_labels_one_hot = np.eye(gallery_length)[probe_labels-136-1]
    # Convert probe_labels to one hot encoding, to identify which subject it belongs to
    # Eg: probe_labels = [6, 84, 14] is converted to [[6 in OHE], [84 in OHE], [14 in OHE]]
    # Each OHE is of size (gallery_length), because it says which gallery sequence it corresponds to
    # probe_labels-136-1 is needed to make sure indices start from 0.


    # Now, similarities and OHE_probe_labels are both of shape (Num_probes, Gallery_length)
    # Compute ROC after flattening it

    similarities = np.reshape(similarities, (-1))
    probe_labels_one_hot = np.reshape(probe_labels_one_hot, (-1))

    fpr,tpr, _ = roc_curve(probe_labels_one_hot, similarities)
    tpr_values_list = find_idx(fpr, tpr)   # TPR values at desired FPR values (1%, 5%, 10%)
    tpr_values_list.append(accuracy)
    return tpr_values_list


def get_all_scores(features, labels, condition_list, session_list): 
    # Take features and labels, compute all the scores for all protocols.
    
    # INPUT: 
    # features: Numpy array of size (num_sequences, embedding) - One embedding for every video in test set
    # labels: Numpy array of size (num_sequences), containing integer labels from {137, 138....226}
    # condition_list: Python list of length num_sequences. Each element is a number from {1,2,3....12} denoting walking condition. 
    # session_list: Python list of length num_sequences, each element being either 1 or 2. Denotes session number of the sequence. 
    # Refer Fig 2 in http://cvlab.cse.msu.edu/frontal-view-gaitfvg-database.html for interpretation of labels annd sessions
    
    # OUTPUT: 
    # scores, a dictionary with keys as 'all', 'ws', 'cl' etc denoting protocol. Each entry in dictionary is a list.
    # Eg. scores['all'] is a list [TPR@1% FPR, TPR@5%, TPR%10%, Accuracy.]
    
    protocols = ['ws', 'all', 'cb', 'cl', 'cbg']
    scores = {}

    for protocol in protocols:
        scores[protocol] = get_scores(protocol=protocol, features=features, labels=labels, condition_list=condition_list, session_list=session_list)

    # scores is a dictionary: Eg. scores['ws'] is a list of 3 elements - TPR at 1%, 5%, 10% FPR
    return scores

