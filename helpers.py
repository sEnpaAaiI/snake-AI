"""
Included various helper functions
"""
import pickle
import numpy as np

def save_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def laod_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    

def load_weights(game_no, model):
    model.l1 = np.load(f"weights/{game_no}/_score_l1.npy")
    model.l1_b = np.load(f"weights/{game_no}/_score_l1_b.npy")

    model.l2 = np.load(f"weights/{game_no}/_score_l2.npy")
    model.l2_b = np.load(f"weights/{game_no}/_score_l2_b.npy")