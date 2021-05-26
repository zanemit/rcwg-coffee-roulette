import sys
import numpy as np
import os
import pickle
from coffee_roulette_config import crConfig

def create_coffee_roulette(working_dir):
    """
    Creates .npy and .pkl files for storing a meeting probability matrix and a
    dictionary of participant names respectively.
    This function only needs to be run once.
    
    PARAMS:
      working_dir (str) : full path to desired file directory
    """
    if os.path.exists(crConfig.file_npy) and os.path.exists(crConfig.file_pkl):
        print("Coffee roulette already exists! Run another function!")
    else:       
        crParticipantDict = {}
        id = 0
        while True:
            participant = input("Enter participant name or 'x' if you are done: ")
            if participant == 'x':
                break
            crParticipantDict[id] = (participant, [])    
            id += 1
        pickle.dump(crParticipantDict, open(crConfig.file_pkl, "wb" ))
        
        crProbMatrix = np.full((id, id), 1/(id-1))  
        np.fill_diagonal(crProbMatrix, 0)
        np.save(crConfig.file_npy, crProbMatrix)
        
def add_participants(working_dir, participants):  
    """
    Adds new participants to the matrix and dictionary.
    ---
    PARAMS:
      working_dir (str) : full path to file directory
      participants (list of str) : names of participants to be added
    """  
    # load files    
    crProbMatrix = np.load(crConfig.file_npy)
    crParticipantDict = pickle.load(open(crConfig.file_pkl, "rb" ))
    
    for name in participants:
        i = crProbMatrix.shape[0]
        
        # update participant matrix
        crProbMatrix = np.hstack((crProbMatrix, np.full((i,1), 1)))
        crProbMatrix = np.vstack((crProbMatrix, np.full((1, i+1), 1)))
        np.fill_diagonal(crProbMatrix, 0)
        
        # update participant dictionary
        crParticipantDict[i] = (name, [])
    
    for i in range(crProbMatrix.shape[0]):
        i_not0 = np.nonzero(crProbMatrix[i,:])[0]
        crProbMatrix[i,:][crProbMatrix[i,:]>0] = 1/len(i_not0)
        
    np.save(crConfig.file_npy, crProbMatrix)
    pickle.dump(crParticipantDict, open(crConfig.file_pkl, "wb" ))
                                
def get_pairs(working_dir, conv_starters = True, odd_nonrandom = True):
    """
    Generates pairs of participants based on the probability matrix
    such that people can only be paired with those they have not had
    coffee with yet.
    Prints the freshly generated coffee date pairs and, in case of an
    odd number of participants, the name of the left-out person.
    ---
    PARAMS:
      working_dir (str) : full path to desired file directory
      conv_starters (bool) : whether to include suggested conversation starters in the output
      odd_nonrandom (bool) : excludes participant #25 if True, but someone randomly otherwise
    """
    # load_files
    crProbMatrix = np.load(crConfig.file_npy)
    crParticipantDict = pickle.load(open(crConfig.file_pkl, "rb" ))
    
    # if odd number of participants, have Zane sit the round out
    if crProbMatrix.shape[0] % 2 == 1 and odd_nonrandom:
        used_ids = np.asarray([25])
    else:    
        used_ids = np.empty(0)
        
    for p in range(int(crProbMatrix.shape[0]/2)):
        unused_ids = np.setdiff1d(np.arange(crProbMatrix.shape[0]), used_ids)
        max_val_remaining_crProbMatrix = crProbMatrix[np.repeat(unused_ids, unused_ids.shape), np.tile(unused_ids, unused_ids.shape)].max()
        i, j = np.where(crProbMatrix == max_val_remaining_crProbMatrix)
        for x in used_ids:
            ids_to_remove = np.where(i == x)
            i = np.delete(i, ids_to_remove)
            j = np.delete(j, ids_to_remove)
            jds_to_remove = np.where(j == x)
            i = np.delete(i,jds_to_remove)
            j = np.delete(j,jds_to_remove)
        id = np.random.choice(len(i))
        if conv_starters == True:
            # find conv starters not used by either participant
            used_conv_starters = crParticipantDict[i[id]][1] + crParticipantDict[j[id]][1]
            all_conv_starters = np.arange(len(open(crConfig.file_txt).readlines( )))
            available_conv_starters = np.setdiff1d(all_conv_starters, used_conv_starters)
            # pick a random question
            if available_conv_starters.shape[0] > 0:
                q_id = np.random.choice(available_conv_starters)
                with open(crConfig.file_txt, "r") as text_file:
                    for l, line in enumerate(text_file):
                        if l == q_id:
                            question = line   
            else:
                print("All conversation starters have been used! Add more!")
                break
            # add the question id to participants' conversation starter history
            crParticipantDict[i[id]][1].append(q_id)
            crParticipantDict[j[id]][1].append(q_id)
            output_str = f"- @{crParticipantDict[i[id]][0]} will have coffee with @{crParticipantDict[j[id]][0]}. Suggested conversation starter: {question} "
        else:    
            output_str = f"- @{crParticipantDict[i[id]][0]} will have coffee with @{crParticipantDict[j[id]][0]}"
        print(output_str)
        used_ids = np.concatenate((used_ids, [i[id], j[id]]))
        crProbMatrix[i[id],j[id]] = 0
        crProbMatrix[j[id],i[id]] = 0
        
    if crProbMatrix.shape[0] % 2 == 1 and not odd_nonrandom:
        unpaired_id = np.setdiff1d(np.arange(crProbMatrix.shape[0]), used_ids)
        print(crParticipantDict[unpaired_id[0]][0], "was not paired with anyone this time!")
    
    # update probability matrix
    for row in range(crProbMatrix.shape[0]):
        nonzeros = crProbMatrix[row] != 0
        crProbMatrix[row,nonzeros] = 1/nonzeros.sum()
    np.save(crConfig.file_npy, crProbMatrix) 
    pickle.dump(crParticipantDict, open(crConfig.file_pkl, "wb" ))
