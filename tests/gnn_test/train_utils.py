"""
This code shows how to perform regression of protonation energy of each molecule with the dataset provided by Romit Maulik,
using a GNN based on edge-conditioned convolutions in batch mode. This specific code contains the training related functions.

Author: Shengli Jiang
Email: sjiang87@wisc.edu / shengli.jiang@anl.gov
"""

import numpy as np
import pickle
import rdkit.Chem as Chem
from Data_Processing import molecule

# define necessary parameters
elem_list = ['C', 'O', 'H', 'unknown']  # species of atoms
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1  # number of atomic features
bond_fdim = 6  # number of bond features
max_nb = 6  # maximum number of neighbors
output_fdim = 1  # output feature dimension
max_atom = 31  # maximum atoms in the molecule


def load_dict(name):
    """
    Load the pickle data file.
    Args:
        name: str, data path

    Returns:
        pickle file of data
    """
    with open(name, 'rb') as f:
        return pickle.load(f)


def load_data_split(train_key_path, valid_key_path):
    """
    Load training and validation keys.
    Args:
        train_key_path: str, training key path
        valid_key_path: str, validation key path

    Returns:
        list of training and validation keys
    """
    train_key_vals = []
    valid_key_vals = []

    with open(train_key_path) as f:
        f.readline()
        for line in f:
            key = line.strip("\n ")
            train_key_vals.append(key)

    with open(valid_key_path) as f:
        f.readline()
        for line in f:
            key = line.strip("\n ")
            valid_key_vals.append(key)

    return train_key_vals, valid_key_vals


def onek_encoding_unk(x, allowable_set):
    """
    Args:
        x: scalar, input value
        allowable_set: list, possible value set

    Returns:
        a list of encoded values
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    """
    Generate an array of atom feature
    Args:
        atom: rdkit atom

    Returns:
        an array of atom feature
    """
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
                    + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                    + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    """
    Generate an arrat of bond feature
    Args:
        bond: rdkit bond

    Returns:
        an array of bond feature
    """
    bt = bond.GetBondType()
    return np.array(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
         bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def dict2graph(key, idict, idxfunc=lambda x: x.GetIdx()):
    """
    Generate graph features
    Args:
        key: str, chemical key name
        idict: dict, data dict
        idxfunc: func, get atom index

    Returns:
        A: Adjacency matrix
        X: Node features
        E: Edge features
        m: Mask for existing atoms
        y: Labels for each atom

    """
    mol = idict[key].rdkit_mol

    if not mol:
        raise ValueError("Could not parse :", key)

    n_atoms = mol.GetNumAtoms()
    n_bonds = max(mol.GetNumBonds(), 1)
    A = np.zeros((max_atom, max_atom))
    X = np.zeros((max_atom, atom_fdim))
    E = np.zeros((max_atom, max_atom, bond_fdim))
    m = np.zeros((max_atom, 1))
    y = np.zeros((max_atom, output_fdim))

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        X[idx, ...] = atom_features(atom)
        A[idx, idx] = 1
        if atom.GetSymbol() != 'C' and atom.GetSymbol() != 'H':
            m[idx, ...] = 1

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        A[a1, a2] = 1
        A[a2, a1] = 1
        E[a1, a2, ...] = bond_features(bond)
        E[a2, a1, ...] = bond_features(bond)

    # Need to process outputs
    y[:n_atoms, ...] = np.reshape(idict[key].prot_energy, newshape=(n_atoms, output_fdim))

    return A, X, E, m, y


def load_data_func(key_list, idict, idxfunc=lambda x: x.GetIdx()):
    """
    Load the data for training or testing.
    Args:
        key_list: list, containing chemical name keys
        idict: dict, all data information
        idxfunc: func, get the index

    Returns:
        A: Adjacency matrix
        X: Node features
        E: Edge features
        m: Mask for existing atoms
        y: Labels for each atom

    """
    res = list(map(lambda x: dict2graph(x, idict, idxfunc), key_list))
    A, X, E, m, y = zip(*res)
    A = np.array(A).squeeze()
    X = np.array(X).squeeze()
    E = np.array(E).squeeze()
    m = np.array(m).squeeze()
    y = np.array(y).squeeze()
    return A, X, E, m, y
