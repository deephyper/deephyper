import numpy as np
import glob, os, sys

HERE = os.getcwd()
PARENT = os.path.dirname(os.getcwd())
BASE = os.path.dirname(PARENT)
sys.path.insert(0,BASE)

# print(PARENT)
# exit()

import warnings
import pickle
import random

# Information required by graph learning
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
import xyz2mol as x2m #To save rdkit mol object as well

warnings.filterwarnings('ignore') #For some future warnings stuff - enable after validation!!
base_path = os.getcwd()

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def data_parser(csv_name,directory_name):
    # Load protonation data (in a sorted manner as the *xyz structures)
    text_file = open(csv_name, "r")
    prot_data = text_file.read().splitlines()

    csv_filename_list = []
    filename_list = []
    csv_energy_list = []
    energy_list = []
    # Check consistency of data
    for line in prot_data:
        csv_filename_list.append(line.split()[0]+str('.xyz'))
        csv_energy_list.append(float(line.split()[1]))

    max_energy = 0
    min_energy = 100

    # The following code gives us the actual data we have
    for i in range(len(csv_filename_list)):
        if os.path.isfile(directory_name+csv_filename_list[i]):
            filename_list.append(csv_filename_list[i])
            energy_list.append(csv_energy_list[i])

            max_energy = max(max_energy,csv_energy_list[i])
            min_energy = min(min_energy,csv_energy_list[i])

    # Now we can use filename_list to find xyz files and protonation locations
    # Energy list can tell us about targets and their locations

    # The following snippet is used to collate experiments into graph samples
    graph_dict = {}
    for i in range(len(filename_list)):
        molecule_name = filename_list[i].split('.')[0].split('_')[0]
        protonation_location = int(filename_list[i].split('.')[0].split('_')[-1])

        if molecule_name not in graph_dict:
            try:
                mol_object = molecule(directory_name+filename_list[i])
                mol_object.prot_energy[protonation_location] = energy_list[i]

                # Read file into memory as well
                reader = open(directory_name+filename_list[i],'r')
                mol_object.lines = reader.readlines()
                reader.close()
        
                graph_dict[molecule_name] = mol_object

            except:
                print('Error!')    
                print(filename_list[i],' has been omitted from data')

        else:
            mol_object.prot_energy[protonation_location] = energy_list[i]

    return graph_dict, min_energy, max_energy


class molecule:
    def __init__(self,xyz_file):
        self.filepath = xyz_file

        atomicNumList, charge, xyz_coordinates = x2m.read_xyz_file(xyz_file)
        charged_fragments = False #False originally
        quick = True

        # RDkit molecule object generation - refer xyz2mol documentation
        self.rdkit_mol = x2m.xyz2mol(atomicNumList,charge,xyz_coordinates,charged_fragments,quick)
        self.prot_energy = np.zeros(shape=self.rdkit_mol.GetNumAtoms())

        # xyz file strings
        self.lines = None


if __name__ == "__main__":
    # # Parse data and save to dictionaries
    # data_dict_1, min_energy_1, max_energy_1 = data_parser('data_1.csv','Structures/')
    data_dict, min_energy_2, max_energy_2 = data_parser('data_2.csv','Structures/')

    min_energy = min(min_energy_1, min_energy_2)
    max_energy = max(max_energy_1, max_energy_2)

    print(min_energy,max_energy)
    
    data_dict.update(data_dict_1)

    # # Normalize protonation energies between 0 and 1 for *.csv files
    # for key in data_dict.keys():
    #     prot_energy_array = np.asarray(data_dict[key].prot_energy[:])
    #     prot_energy_array = np.where(np.abs(prot_energy_array)>0.01,(prot_energy_array-min_energy)/(max_energy-min_energy),prot_energy_array)
    #     prot_energy_array = (prot_energy_array-min_energy)/(max_energy-min_energy)
    #     data_dict[key].prot_energy[:] = prot_energy_array[:]

    # Split into train and test dictionaries - randomizing
    dict_list = list(data_dict.items())
    random.shuffle(dict_list)

    train_dict = dict(dict_list[:int(2*len(data_dict)/3)])
    valid_dict = dict(dict_list[int(2*len(data_dict)/3):])
    
    f = open('Train_Keys.txt','w')
    for key in train_dict.keys():
        f.write(key+'\n')
    f.close()

    f = open('Valid_Keys.txt','w')
    for key in valid_dict.keys():
        f.write(key+'\n')
    f.close()

    save_obj(data_dict,'Total_Data')
    