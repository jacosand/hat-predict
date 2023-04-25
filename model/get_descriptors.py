#!/usr/bin/env python

"""
Script to compute all DFT descriptors needed as input to the ML model to
predict the free energy barriers for a list of hydrogen atom transfer (HAT)
reactions specified in a .csv file containing the SMILES strings of all
species in each reaction (default: smiles.csv).  Writes descriptors to
another .csv file (default: descriptors.csv).  If the input file contains
ground-truth free energy barriers, these barriers are also written to the
output file.

Run from the command line:
python get_descriptors.py [arguments] /path/to/directory/with/smiles.csv

For possible arguments, see argument parser in __main__ section or run:
python get_descriptors.py -h
"""

import os
import argparse
from functools import lru_cache

import csv
import pandas as pd

from rdkit import Chem

from morfeus import BuriedVolume, read_xyz

import requests
import urllib.parse
from bs4 import BeautifulSoup


class Molecule:
    def __init__(self, smiles):
        """
        Initializes a Molecule object with specified SMILES string

        Arguments:

            smiles (str): SMILES string representing the molecule

        Attributes:

            smiles (str): SMILES string of the molecule

            elements (list): list of atomic symbols of the atoms
            coords (list): list of 3D Cartesian coordinates of the atoms
        
            HOMO (float): HOMO energy
            LUMO (float): LUMO energy
            energy (float): total energy

            IP (float): ionization potential
            EA (float): electron affinity

            charges (list): list of atomic charges of the atoms

            bondsH (list): list of Wiberg bond orders for each atom to H atoms
            bondsX (list): list of Wiberg bond orders for each atom to non-H atoms

            fukui_p (list): list of f(+) functions for each atom
            fukui_0 (list): list of f(0) functions for each atom
            fukui_m (list): list of f(-) functions for each atom
        
        Note that the ordering of atoms is the same in all of the lists.
        """

        self.smiles = smiles

        self.elements = []
        self.coords = []
        
        self.HOMO = None
        self.LUMO = None
        self.energy = None
        self.IP = None
        self.EA = None
        
        self.charges = []

        self.bondsH = []
        self.bondsX = []

        self.fukui_p = []
        self.fukui_0 = []
        self.fukui_m = []
    
    def compute_xtb_features(self, name, charge, unpaired, scratch_dir, opt):
        """
        Uses xTB quantum chemistry package to compute various molecular features
        such as optimized geometry, orbital energies, molecular energy, ionization
        potential, electron affinity, charges, Wiberg bond orders, and Fukui
        functions.
        
        These features are updated in the Molecule object.

        Arguments:

            name (str): base name of .xyz file containing molecular structure

            charge (int): total charge of the molecule

            unpaired (int): total number of unpaired electrons in the molecule
        
            scratch_dir (str): path to directory where temporary files will be
                               stored during computation of chemical descriptors
                               with xTB

            opt (bool): whether to perform geometry optimization with xTB
                        (usually True, but False for single-atom radicals)

        Returns:

            self (Molecule): Molecule object with updated features
        """

        c = str(charge)
        u = str(unpaired)

        current_dir = os.getcwd()
        
        os.chdir(scratch_dir)

        # Run xTB geometry optimization, then remove temporary files
        # Handle single-atom radicals as a special case
        if opt == False:
            os.system(f"xtb {name}.xyz --chrg {c} --uhf {u} --namespace {name} > {name}.opt 2> /dev/null")
            os.system(f"cp {name}.xyz {name}.xtbopt.xyz")
            os.system(f"rm {name}.xtbrestart")
        else:
            os.system(f"xtb {name}.xyz --chrg {c} --uhf {u} --namespace {name} --opt > {name}.opt 2> /dev/null")
            os.system(f"rm {name}.xtbrestart {name}.xtbtopo.mol {name}.xtbopt.log .{name}.xtboptok")
        
        # Run xTB IP/EA calculation starting from optimized geometry, then remove temporary files
        os.system(f"xtb {name}.xtbopt.xyz --chrg {c} --uhf {u} --namespace {name}_ipea --vipea > {name}.ipea  2> /dev/null")
        if opt == False:
            os.system(f"rm {name}_ipea.charges {name}_ipea.wbo {name}_ipea.xtbrestart")
        else:
            os.system(f"rm {name}_ipea.charges {name}_ipea.wbo {name}_ipea.xtbrestart {name}_ipea.xtbtopo.mol")

        # Run xTB Fukui calculation starting from optimized geometry, then remove temporary files
        # Handle single-atom radicals as a special case with known Fukui functions
        if opt == False:
            with open(f"{name}.fukui", 'w', encoding='utf-8') as f:
                f.write('     #        f(+)     f(-)     f(0)\n')
                f.write('     1X      -1.000   -1.000   -1.000\n')
                f.write('           -------------------------------------------------\n')
        else:
            os.system(f"xtb {name}.xtbopt.xyz --chrg {c} --uhf {u} --namespace {name}_fukui --vfukui > {name}.fukui  2> /dev/null")
            os.system(f"rm {name}_fukui.charges {name}_fukui.wbo {name}_fukui.xtbrestart {name}_fukui.xtbtopo.mol")

        # Read optimized geometry
        self.elements, self.coords = read_xyz(f"{name}.xtbopt.xyz")

        # Read orbital energies and total energy from xTB geometry optimization
        start_orbitals = False
        with open(f"{name}.opt", encoding='utf-8') as f:
            for line in f:
                # Fix xTB lines that contain non-UTF-8 characters
                line = line.encode('utf-8', errors='replace').decode('utf-8')
                if line.strip() == '* Orbital Energies and Occupations':
                    start_orbitals = True
                # HOMO energy
                elif start_orbitals and line.strip().endswith('(HOMO)'):
                    words = line.split()
                    self.HOMO = float(words[-2])
                # LUMO energy
                elif start_orbitals and line.strip().endswith('(LUMO)'):
                    words = line.split()
                    self.LUMO = float(words[-2])
                # Total energy
                elif start_orbitals and line.strip().startswith('| TOTAL ENERGY'):
                    words = line.split()
                    self.energy = float(words[3])
    
        # Read charges from end of xTB geometry optimization
        with open(f"{name}.charges", encoding='utf-8') as f:
            self.charges = f.read().split()

        # Read Wiberg bond orders for each atom from end of xTB geometry optimization
        self.bondsH = [[] for _ in range(len(self.elements))]
        self.bondsX = [[] for _ in range(len(self.elements))]

        with open(f"{name}.wbo", encoding='utf-8') as f:
            # Each line represents a potential bond
            for line in f:
                words = line.split()
                # Bond detected if bond order > 0.5
                if float(words[2]) > 0.5:
                    a1_index = int(words[0])-1
                    a2_index = int(words[1])-1
                    bond_length = float(words[2])

                    # Add bond length to list of bonds for atom 1
                    if self.elements[a2_index] == 'H':
                        self.bondsH[a1_index].append(bond_length)
                    else:
                        self.bondsX[a1_index].append(bond_length)

                    # Add bond length to list of bonds for atom 2
                    if self.elements[a1_index] == 'H':
                        self.bondsH[a2_index].append(bond_length)
                    else:
                        self.bondsX[a2_index].append(bond_length)

        # Read IP and EA from xTB IP/EA calculation
        with open(f"{name}.ipea", encoding='utf-8') as f:
            for line in f:
                # IP
                if line.strip().startswith('delta SCC IP (eV)'):
                    words = line.split()
                    self.IP = float(words[-1])
                # EA
                elif line.strip().startswith('delta SCC EA (eV)'):
                    words = line.split()
                    self.EA = float(words[-1])

        # Read Fukui functions for each atom from xTB Fukui calculation
        start_fukui = False
        with open(f"{name}.fukui", encoding='utf-8') as f:
            # Each line after the header represents an atom
            for line in f:
                if line.strip() == '#        f(+)     f(-)     f(0)':
                    start_fukui = True
                elif start_fukui and line.strip().startswith('--'):
                    break
                elif start_fukui:
                    words = line.split()
                    # f+
                    self.fukui_p.append(float(words[1]))
                    # f-
                    self.fukui_m.append(float(words[2]))
                    # f0
                    self.fukui_0.append(float(words[3]))

        # Remove files that were read
        os.system(f"rm {name}.xtbopt.xyz {name}.opt {name}.charges {name}.wbo {name}.ipea {name}.fukui")

        os.chdir(current_dir)

        return self

    def get_features_on_atom(self, atom_index):
        """
        Function which looks up and returns a dictionary of all
        features on a given atom_index.  Features pertaining to the
        molecule as a whole are returned too.

        Buried volumes are computed for the specified atom_index and
        returned in the feature dictionary.

        Arguments:

            index (int): index of the atom for which to look up features
                         and compute buried volumes

        Returns:

            features (dict): dictionary containing all features which pertain
                             to the atom_index or the molecule as a whole
        """

        atom_number = atom_index + 1

        features = {}

        features['HOMO'] = self.HOMO
        features['LUMO'] = self.LUMO
        features['energy'] = self.energy
        features['IP'] = self.IP
        features['EA'] = self.EA

        features['charge'] = self.charges[atom_index]
        features['f+'] = self.fukui_p[atom_index]
        features['f0'] = self.fukui_0[atom_index]
        features['f-'] = self.fukui_m[atom_index]

        features['bondsH'] = self.bondsH[atom_index]
        features['bondsX'] = self.bondsX[atom_index]
        features['bonds'] = features['bondsH'] + features['bondsX']

        # Compute buried volumes for the atom using morfeus.
        # Note that the morfeus API uses one-indexed atom numbers.
        bv3 = BuriedVolume(self.elements, self.coords, atom_number, include_hs=True, radius=3, radii_scale=1).fraction_buried_volume
        bv4 = BuriedVolume(self.elements, self.coords, atom_number, include_hs=True, radius=4, radii_scale=1).fraction_buried_volume
        bv5 = BuriedVolume(self.elements, self.coords, atom_number, include_hs=True, radius=5, radii_scale=1).fraction_buried_volume
        features['bv3'] = bv3
        features['bv4'] = bv4
        features['bv5'] = bv5

        return features


@lru_cache(maxsize=None)
def compute_molecule(smiles, name, charge, unpaired, scratch_dir, opt):
    """
    Convenience function which computes all features for a molecule and
    caches the result.

    Arguments:

        smiles (str): SMILES string for the molecule

        name (str): base name of the molecule to use for filenames
                    in xTB calculations

        charge (int): total charge of the molecule

        unpaired (int): total number of unpaired electrons in the molecule

        scratch_dir (str): path to directory where temporary files will be
                           stored during computation of chemical descriptors
                           with xTB

        opt (bool): whether to perform geometry optimization with xTB
                    (usually True, but False for single-atom radicals)

    Returns:

        molecule (Molecule): Molecule object containing all computed features
    """

    return Molecule(smiles).compute_xtb_features(name, charge, unpaired, scratch_dir, opt)


@lru_cache(maxsize=None)
def get_atom(molecule, atom_index):
    """
    Convenience function which returns a dictionary of all features
    on a given atom_index, and caches the result.

    Arguments:

        molecule (Molecule): Molecule object containing all computed features
    
        atom_index (int): index of the atom for which to look up features

    Returns:

        features (dict): dictionary containing all features which pertain
                         to the atom_index
    """
    
    return molecule.get_features_on_atom(atom_index)


@lru_cache(maxsize=None)
def query_alfabet(smiles):
    """
    Makes a web request that queries the alfabet GCN model for the BDEs and
    BDFEs of all bonds to H in a given molecule.  The web request is parsed
    into a dictionary whose keys are the SMILES strings of the radicals, and
    whose values are tuples of the (BDE, BDFE) for removing the H atom to
    generate that radical.  Results are cached.

    Arguments:
    
        smiles (str): SMILES string for the molecule to query
    
    Returns:

        out (dict): dictionary whose keys are the SMILES strings of all radicals
                    that can be formed by removing an H atom from the molecule,
                    and whose values are tuples of floats (BDE, BDFE) for
                    generating that radical
    """

    # Handle a few special radicals which alfabet does not accept by directly
    # providing the computed values that alfabet would attempt to predict
    if smiles == 'F':
        return {'[F]': (132.5, 126.0)}
    if smiles == 'Cl':
        return {'[Cl]': (101.6, 95.5)}
    if smiles == 'Br':
        return {'[Br]': (86.3, 80.2)}
    if smiles == 'I':
        return {'[I]': (69.1, 63.2)}
    if smiles == 'C1C[NH+]2CCC1CC2':
        return {'C1C[N+]2CCC1CC2': (96.6, 87.8)}

    # Make web request and parse HTML using BeautifulSoup
    url = f"https://bde.ml.nrel.gov/result?name={urllib.parse.quote(smiles)}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    out = {}

    # Generate RDKit Mol objects for molecule with and without explicit Hs
    mol = Chem.MolFromSmiles(smiles)
    mol_explicitH = Chem.AddHs(Chem.MolFromSmiles(smiles))

    # Loop over all bonds whose BDE and BDFE were computed by alfabet
    for bond in soup.find_all('div', class_='media-body'):
        bond_type = bond.find('strong', string='Bond Type:').next_sibling.strip()

        # Only process bonds to H
        if bond_type.startswith('H') or bond_type.endswith('H'):

            # Extract BDE and BDFE
            bde = float(bond.find('strong', string='BDE(ML):').next_sibling.text.split(' ')[1])
            bdfe = float(bond.find('strong', string='BDFE(ML):').next_sibling.text.split(' ')[1])

            # Extract bond index which corresponds to bond index in mol_explicitH
            bond_index = int(bond.find('a', class_='neighbor-link')['href'].split('=')[-1])

            # Get atom_idx of heavy atom in bond
            atom_idx = mol_explicitH.GetBondWithIdx(bond_index).GetBeginAtomIdx()

            # Get atom object in mol corresponding to atom_idx
            atom = mol.GetAtomWithIdx(atom_idx)

            rcount = atom.GetNumRadicalElectrons()
            hcount = atom.GetTotalNumHs()

            # Convert mol object to radical by removing H atom
            atom.SetNumExplicitHs(hcount - 1)
            atom.SetNumRadicalElectrons(rcount + 1)

            # Generate radical SMILES string and add to dictionary
            smilesR = Chem.MolToSmiles(mol)
            out[smilesR] = (bde, bdfe)

            # Convert mol object back to original molecule
            atom.SetNumExplicitHs(hcount)
            atom.SetNumRadicalElectrons(rcount)
    
    return out


@lru_cache(maxsize=None)
def get_optimized_geometry(smiles, name, charge, unpaired, scratch_dir):
    """
    This function returns the geometry of the lowest energy conformer of a
    molecule with the specified SMILES string.  The lowest energy conformer is
    optimized with the xTB quantum chemistry package and returned as a string
    containing the content of the .xyz file for that geometry.  Results are
    cached.

    Arguments:

        smiles (str): SMILES string for the molecule

        name (str): base name of the molecule to use for filenames
                    in xTB calculations

        charge (int): total charge of the molecule

        unpaired (int): total number of unpaired electrons in the molecule

        scratch_dir (str): path to directory where temporary files will be
                           stored during computation of chemical descriptors
                           with xTB
        
    Returns:

        xyz (str): content of the .xyz file for the lowest energy conformer
                   of the molecule
    """

    c = str(charge)
    u = str(unpaired)

    current_dir = os.getcwd()
    
    os.chdir(scratch_dir)

    # Perform conformational search with OpenBabel and write lowest conformaton to .xyz file
    os.system(f"obabel -:\"{smiles}\" -O {name}.xyz --gen3d --ff MMFF94 > /dev/null 2>&1")

    # Run xTB geometry optimization
    os.system(f"xtb {name}.xyz --chrg {c} --uhf {u} --namespace {name} --opt > {name}.opt  2> /dev/null")

    # Read optimized geometry from .xtbopt.xyz file
    with open(f"{name}.xtbopt.xyz", encoding='utf-8') as f:
        xyz = f.read()

    # Remove temporary files
    os.system(f"rm {name}.xtbrestart {name}.xtbtopo.mol {name}.xtbopt.log .{name}.xtboptok")
    os.system(f"rm {name}.xyz {name}.xtbopt.xyz {name}.opt")

    os.chdir(current_dir)

    return xyz


@lru_cache(maxsize=None)
def molecule_pair_descriptors(smilesR, nameR, smilesH, nameH, atom_index, scratch_dir):
    """
    Takes as input a radical and a corresponding hydrogenated species, each
    with a specified atom index, and returns two dictionaries containing
    calculated descriptors for the radical and the hydrogenated species.
    
    The alfabet model is used to compute the BDE and BDFE to convert the
    hydrogenated species into the radical, and the descriptors are added
    to the radical dictionary.

    Arguments:

        smilesR (str): SMILES string of the radical

        atomR (int): atom index specifying at which atom in the radical
                     to compute atom-specific descriptors

        nameR (str): base name to use for filenames in xTB calculations
                     performed on the radical

        smilesH (str): SMILES string of the hydrogenated species corresponding
                       to the radical
            
        atom_index (int): atom index specifying at which atom to compute
                          atom-specific descriptors
                          (Note: Because the radical species is generated here by
                          removing a hydrogen atom from the hydrogenated species,
                          and heavy atoms always appear above hydrogen atoms, the
                          atom index we want is the same for both species.)
            
        nameH (str): base name to use for filenames in xTB calculations
                     performed on the hydrogenated species

        scratch_dir (str): path to directory where temporary files will be
                           stored during computation of chemical descriptors
                           with xTB
    
    Returns:

        molR (dict): dictionary containing calculated descriptors
                     for the radical

        molH (dict): dictionary containing the calculated descriptors
                     for the hydrogenated species
    """

    # Compute net charge of each species
    qR = Chem.rdmolops.GetFormalCharge(Chem.MolFromSmiles(smilesR))
    qH = Chem.rdmolops.GetFormalCharge(Chem.MolFromSmiles(smilesH))

    # Get optimized geometry of the hydrogenated species using xTB
    xyzH = get_optimized_geometry(smilesH, nameH, qH, 0, scratch_dir)

    current_dir = os.getcwd()
    os.chdir(scratch_dir)

    # Write optimized geometry of hydrogenated species to .xyz file
    with open(f"{nameH}.xyz", 'w', encoding='utf-8') as f:
        f.write(xyzH)

    # Find the atom index of a hydrogen atom in the hydrogenated species
    # which could be removed to generate the radical species
    for atom in Chem.AddHs(Chem.MolFromSmiles(smilesH)).GetAtomWithIdx(int(atom_index)).GetNeighbors():
        if atom.GetSymbol() == 'H':
            remove_idx = atom.GetIdx()
            break

    # Write new .xyz file for the radical using the optimized geometry
    # of the hydrogenated species but with the identified hydrogen atom removed
    # (Note: This geometry will be reoptimized with xTB in compute_molecule.)
    with open(f"{nameR}.xyz", 'w', encoding='utf-8') as f:
        for i, line in enumerate(xyzH.split('\n')):
            # Reduce number of atoms by 1
            if i == 0:
                f.write(f"{int(line.strip()) - 1}\n")
            # Write line unless it corresponds to identified hydrogen atom
            elif i-2 != remove_idx:
                f.write(f"{line}\n")

    os.chdir(current_dir)

    # Obtain chemical descriptors for each species at specified atom index
    molH = get_atom(compute_molecule(smilesH, nameH, qH, 0, scratch_dir, True), atom_index)

    # Catch single-atom radical species that do not require geometry optimization
    if smilesR in {"[F]", "[Cl]", "[Br]", "[I]"}:
        molR = get_atom(compute_molecule(smilesR, nameR, qR, 1, scratch_dir, False), atom_index)
    else:
        molR = get_atom(compute_molecule(smilesR, nameR, qR, 1, scratch_dir, True), atom_index)

    os.chdir(scratch_dir)
    os.system(f"rm {nameH}.xyz {nameR}.xyz")
    os.chdir(current_dir)

    # Query alfabet for BDE and BDFE to convert hydrogenated species into radical
    molR['bde'], molR['bdfe'] = query_alfabet(smilesH)[smilesR]

    return molR, molH


def reaction_descriptors(BR_desc, BH_desc, AR_desc, AH_desc):
    """
    Function which computes all DFT descriptors needed for the input of one
    hydrogen atom transfer (HAT) reaction of the form
    A•   +  B-H   -->   A-H  +   B•  
    into the ML model, returning them as a dictionary.

    Arguments:
    
        BR_desc (dict): dictionary containing all chemical descriptors for B•

        BH_desc (dict): dictionary containing all chemical descriptors for B-H

        AR_desc (dict): dictionary containing all chemical descriptors for A•

        AH_desc (dict): dictionary containing all chemical descriptors for A-H

    Returns:

        desc (dict): a dictionary containing all chemical descriptors needed
                     for the input of one hydrogen atom transfer (HAT) reaction
                     into the ML model.  See function body for all the descriptors
                     and how they are calculated.
    """

    desc = {}

    # Compute bond dissociation energy (DE), enthalpy (DH), and free energy (DG)
    # descriptors in kcal/mol

    HARTREE_TO_KCAL = 627.509469

    desc['DE'] = (BR_desc['energy'] + AH_desc['energy'] - BH_desc['energy'] - AR_desc['energy']) * HARTREE_TO_KCAL

    try:
        desc['DH'] = BR_desc['bde'] - AR_desc['bde']
        desc['DG'] = BR_desc['bdfe'] - AR_desc['bdfe']
    except:
        print('Failed')

    # Compute all electronic descriptors
    # ... for A•
    desc['AR_SOMO'] = AR_desc['HOMO']                             # HOMO
    desc['AR_IE'] = AR_desc['IP']                                 # ionization energy
    desc['AR_EA'] = AR_desc['EA']                                 # electron affinity
    desc['AR_Eneg'] = (desc['AR_IE'] + desc['AR_EA']) / 2         # electronegativity
    desc['AR_Soft'] = 1 / (desc['AR_IE'] - desc['AR_EA'])         # softness

    # ... for A-H
    desc['AH_HOMO'] = AH_desc['HOMO']                             # HOMO
    desc['AH_LUMO'] = AH_desc['LUMO']                             # LUMO
    desc['AH_CP'] = (AH_desc['HOMO'] + AH_desc['LUMO']) / 2       # chemical potential
    desc['AH_Hard'] = (AH_desc['LUMO'] - AH_desc['HOMO']) / 2     # hardness
    desc['AH_Soft'] = 1 / desc['AH_Hard']                         # softness
    desc['AH_Ephil'] = desc['AH_CP'] ** 2 / (2 * desc['AH_Hard']) # electrophilicity

    # ... for B•
    desc['BR_SOMO'] = BR_desc['HOMO']                             # HOMO
    desc['BR_IE'] = BR_desc['IP']                                 # ionization energy
    desc['BR_EA'] = BR_desc['EA']                                 # electron affinity
    desc['BR_Eneg'] = (desc['BR_IE'] + desc['BR_EA']) / 2         # electronegativity
    desc['BR_Soft'] = 1 / (desc['BR_IE'] - desc['BR_EA'])         # softness

    # ... for B-H
    desc['BH_HOMO'] = BH_desc['HOMO']                             # HOMO
    desc['BH_LUMO'] = BH_desc['LUMO']                             # LUMO
    desc['BH_CP'] = (BH_desc['HOMO'] + BH_desc['LUMO']) / 2       # chemical potential
    desc['BH_Hard'] = (BH_desc['LUMO'] - BH_desc['HOMO']) / 2     # hardness
    desc['BH_Soft'] = 1 / desc['BH_Hard']                         # softness
    desc['BH_Ephil'] = desc['BH_CP'] ** 2 / (2 * desc['BH_Hard']) # electrophilicity


    # Compute all charge descriptors
    # ... for A•
    desc['AR_Q'] = AR_desc['charge']                              # Charge
    desc['AR_Fuk0'] = AR_desc['f0']                               # Fukui f0

    # ... for A-H
    desc['AH_Q'] = AH_desc['charge']                              # Charge
    desc['AH_FukP'] = AH_desc['f+']                               # Fukui f+
    desc['AH_FukM'] = AH_desc['f-']                               # Fukui f-
    desc['AH_FukD'] = desc['AH_FukP'] - desc['AH_FukM']           # Fukui Df

    # ... for B•
    desc['BR_Q'] = BR_desc['charge']                              # Charge
    desc['BR_Fuk0'] = BR_desc['f0']                               # Fukui f0

    # ... for B-H
    desc['BH_Q'] = BH_desc['charge']                              # Charge
    desc['BH_FukP'] = BH_desc['f+']                               # Fukui f+
    desc['BH_FukM'] = BH_desc['f-']                               # Fukui f-
    desc['BH_FukD'] = desc['BH_FukP'] - desc['BH_FukM']           # Fukui Df


    # Compute all bond order descriptors
    # ... for A•
    if len(AR_desc['bonds']) == 0:                                          # No bonds (one-atom radical)
        desc['AR_BO'] = 1.205                                               # Impute mean of training data
    else:
        desc['AR_BO'] = sum(AR_desc['bonds']) / len(AR_desc['bonds'])       # Mean bond order

    # ... for A-H
    desc['AH_BO'] = sum(AH_desc['bonds']) / len(AH_desc['bonds'])           # Mean bond order
    desc['AH_BOH'] = sum(AH_desc['bondsH']) / len(AH_desc['bondsH'])        # Mean bond order to H

    if len(AH_desc['bondsX']) == 0:                                         # No bonds to non-H
        desc['AH_BOX'] = 1.003                                              # Impute mean of training data
    else:
        desc['AH_BOX'] = sum(AH_desc['bondsX']) / len(AH_desc['bondsX'])    # Mean bond order to non-H

    # ... for B•
    if len(BR_desc['bonds']) == 0:                                          # No bonds (one-atom radical)
        desc['BR_BO'] = 1.174                                               # Impute mean of training data
    else:
        desc['BR_BO'] = sum(BR_desc['bonds']) / len(BR_desc['bonds'])       # Mean bond order

    # ... for B-H
    desc['BH_BO'] = sum(BH_desc['bonds']) / len(BH_desc['bonds'])           # Mean bond order
    desc['BH_BOH'] = sum(BH_desc['bondsH']) / len(BH_desc['bondsH'])        # Mean bond order to H

    if len(BH_desc['bondsX']) == 0:                                         # No bonds to non-H
        desc['BH_BOX'] = 1.074                                              # Impute mean of training data
    else:
        desc['BH_BOX'] = sum(BH_desc['bondsX']) / len(BH_desc['bondsX'])    # Mean bond order to non-H


    # Compute all steric parameters (buried volume)
    desc['AR_BV3'], desc['AR_BV4'], desc['AR_BV5'] = AR_desc['bv3'], AR_desc['bv4'], AR_desc['bv5']
    desc['AH_BV3'], desc['AH_BV4'], desc['AH_BV5'] = AH_desc['bv3'], AH_desc['bv4'], AH_desc['bv5']
    desc['BR_BV3'], desc['BR_BV4'], desc['BR_BV5'] = BR_desc['bv3'], BR_desc['bv4'], BR_desc['bv5']
    desc['BH_BV3'], desc['BH_BV4'], desc['BH_BV5'] = BH_desc['bv3'], BH_desc['bv4'], BH_desc['bv5']

    return desc


def write_reaction_descriptors(in_file, out_file, scratch_dir, verbose=False):
    """
    This function reads a .csv in_file whose rows contain the SMLIES strings
    and reacting atom indices for all four species in a series of HAT
    reactions, each of the form:
    A•   +  B-H   -->   A-H  +   B•
    
    The computed chemical descriptors are saved to a new .csv out_file.
    For compatability when being used in training, if the in_file contains
    ground-truth free energy barriers, these barriers are also written to out_file.
    
    Arguments:

        in_file (str): path and name of the input .csv file containing the
                       SMILES strings and reacting atom indices for all four
                       species in every HAT reaction

        out_file (str): path and name of the output .csv file to save the computed
                        chemical descriptors for each reaction

        scratch_dir (str): path to directory where temporary files will be stored
                           during computation of chemical descriptors with xTB

        verbose (bool): boolean flag to indicate whether to print out progress in
                        computing descriptors for the reactions to the console

    Returns:

        Nothing, but saves the computed chemical descriptors to the .csv
        out_file.
    """

    field_names = [
        # Reaction label and free energy barrier to be predicted (if present)
        'reaction', 'DGdd_true',

        # SMILES strings and reacting atom indices for all four species
        'AR_smiles', 'AR_atom',
        'AH_smiles', 'AH_atom',
        'BR_smiles', 'BR_atom',
        'BH_smiles', 'BH_atom', 

        # Reaction energy, enthalpy, and free energy
        'DE', 'DH', 'DG',

        # Descriptors for A•
        'AR_SOMO', 'AR_IE', 'AR_EA', 'AR_Eneg', 'AR_Soft',               # electronic descriptors
        'AR_Q', 'AR_Fuk0',                                               # charge descriptors
        'AR_BO',                                                         # bond order descriptors
        'AR_BV3', 'AR_BV4', 'AR_BV5',                                    # steric descriptors

        # Descriptors for A-H
        'AH_HOMO', 'AH_LUMO', 'AH_CP', 'AH_Hard', 'AH_Soft', 'AH_Ephil', # electronic descriptors
        'AH_Q', 'AH_FukP', 'AH_FukM', 'AH_FukD',                         # charge descriptors
        'AH_BO', 'AH_BOH', 'AH_BOX',                                     # bond order descriptors
        'AH_BV3', 'AH_BV4', 'AH_BV5',                                    # steric descriptors

        # Descriptors for B•
        'BR_SOMO', 'BR_IE', 'BR_EA', 'BR_Eneg', 'BR_Soft',               # electronic descriptors
        'BR_Q', 'BR_Fuk0',                                               # charge descriptors
        'BR_BO',                                                         # bond order descriptors
        'BR_BV3', 'BR_BV4', 'BR_BV5',                                    # steric descriptors

        # Descriptors for B-H
        'BH_HOMO', 'BH_LUMO', 'BH_CP', 'BH_Hard', 'BH_Soft', 'BH_Ephil', # electronic descriptors
        'BH_Q', 'BH_FukP', 'BH_FukM', 'BH_FukD',                         # charge descriptors
        'BH_BO', 'BH_BOH', 'BH_BOX',                                     # bond order descriptors
        'BH_BV3', 'BH_BV4', 'BH_BV5',                                    # steric descriptors
    ]

    # Open and write header to .csv output file.
    csv_file = open(out_file, 'w')
    writer = csv.DictWriter(csv_file, fieldnames=field_names, extrasaction='ignore')
    writer.writeheader()

    # Read in .csv file containing HAT reactions to be processed
    reactions = pd.read_csv(in_file)

    if verbose:
        print(f"Processing reactions from {in_file}")

    # Loop over all reactions in the .csv file
    for i in range(reactions.shape[0]):

        if verbose and i % 10 == 0:
            print(f"Computing input descriptors for reaction {i} of {reactions.shape[0]}")

        AR_smiles = reactions['AR_smiles'].iloc[i]
        AR_atom = reactions['AR_atom'].iloc[i]
        AH_smiles = reactions['AH_smiles'].iloc[i]
        AH_atom = reactions['AH_atom'].iloc[i]
        BR_smiles = reactions['BR_smiles'].iloc[i]
        BR_atom = reactions['BR_atom'].iloc[i]
        BH_smiles = reactions['BH_smiles'].iloc[i]
        BH_atom = reactions['BH_atom'].iloc[i]

        # Get descriptors for each species in the reaction
        AR_desc, AH_desc = molecule_pair_descriptors(AR_smiles, 'AR', AH_smiles, 'AH', AH_atom, scratch_dir)
        BR_desc, BH_desc = molecule_pair_descriptors(BR_smiles, 'BR', BH_smiles, 'BH', BH_atom, scratch_dir)

        # Assemble descriptors for the reaction overall
        descriptors = reaction_descriptors(BR_desc, BH_desc, AR_desc, AH_desc)

        # Add reaction label, and ground-truth barrier, to descriptors
        descriptors['reaction'] = reactions['reaction'].iloc[i]
        descriptors['DGdd_true'] = reactions['DGdd_true'].iloc[i]

        # Add SMILES strings and reacting atom indices to descriptors
        descriptors['AR_smiles'] = AR_smiles
        descriptors['AR_atom'] = AR_atom
        descriptors['AH_smiles'] = AH_smiles
        descriptors['AH_atom'] = AH_atom
        descriptors['BR_smiles'] = BR_smiles
        descriptors['BR_atom'] = BR_atom
        descriptors['BH_smiles'] = BH_smiles
        descriptors['BH_atom'] = BH_atom

        # Write reaction descriptors to .csv file
        writer.writerow(descriptors)

    if verbose:
        print(f'Done!  Reaction descriptors written to {out_file}\n')

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Computes chemical descriptors \
        used in the machine learning model for all reactions described in the \
        .csv input file (default name: smiles.csv).  Each line in the .csv output \
        file (default name: descriptors.csv) corresponds to one line in the input \
        file, and contains all chemical descriptors needed for the machine \
        learning model to predict the free energy barrier of that reaction.")

    parser.add_argument(
        "dir",
        help="path to directory which contains .csv file containing SMILES \
              strings of HAT reactions to be computed",
    )

    parser.add_argument(
        "-i", "--inp",
        help="base filename of .csv file (without extension) in which SMILES \
              strings of HAT reactions to be computed are stored",
        default='smiles',
    )

    parser.add_argument(
        "-o", "--out",
        help="base filename of .csv file (without extension) in which chemical \
              descriptors are to be written",
        default='descriptors',
    )

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(
            'Dataset directory %s does not exist!' % args.dir)

    dir_smiles_csv = os.path.join(args.dir, args.inp + '.csv')
    dir_descriptors_csv = os.path.join(args.dir, args.out + '.csv')

    write_reaction_descriptors(in_file = dir_smiles_csv,
                               out_file = dir_descriptors_csv,
                               scratch_dir = args.dir,
                               verbose = True)
