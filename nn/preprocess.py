# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from collections import Counter

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Convert lists to numpy arrays
    seqs = np.array(seqs)
    labels = np.array(labels)

    # Get counts of each class
    label_counts = Counter(labels)
    max_class_size = max(label_counts.values())

    # Separate sequences by class
    pos_seqs = seqs[labels == True]
    neg_seqs = seqs[labels == False]

    # Sample with replacement to balance class sizes
    pos_samples = np.random.choice(pos_seqs, max_class_size, replace=True)
    neg_samples = np.random.choice(neg_seqs, max_class_size, replace=True)

    # Combine sampled sequences and labels
    sampled_seqs = np.concatenate([pos_samples, neg_samples])
    sampled_labels = np.array([True] * max_class_size + [False] * max_class_size)

    return sampled_seqs.tolist(), sampled_labels.tolist()

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    nucleotide_map = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1]
    }

    # Encode each sequence
    encoded_seqs = [
        np.concatenate([nucleotide_map[nuc] for nuc in seq]) for seq in seq_arr
    ]

    return np.array(encoded_seqs)