from django.shortcuts import render
from django.conf import settings

import pandas as pd
import numpy as np

import os
import sys

sys.path.append('../model')
from model.get_descriptors import write_reaction_descriptors
from model.get_barriers import write_reaction_barriers

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

import markdown2


class Radical:
    """
    A class representing a radical species involved in an HAT reaction.
    An HAT reaction is a reaction between two Radical objects as follows:
    A•   +  B-H   -->   A-H  +   B•
    where A• and B• are each represented by a Radical object.

    Attributes:
    - smilesR (str): SMILES string of the radical species
                     (e.g. A• or B•)
    - smilesH (str): SMILES string of the corresponding hydrogenated species
                     (e.g. A-H or B-H)
    - atomR (int): atom number (in RDKit Mol object) of radical species
                   containing the radical
    - atomH (int): atom number (in RDKit Mol object) of hydrogenated species
                   that loses an H atom to form radical
    - hcount (int): number of distinct hydrogen atoms which can be removed
                    to form the radical

    - DGdd (float): ML-computed barrier of the HAT reaction in kcal/mol,
                    typically stored with B•, and obtained by calling
                    compute_barriers

    - select (float): Selectivity of the HAT reaction in % compared to other
                      possible B• radicals which could be produced, typically
                      stored with B•, and obtained by calling compute_selectivities

    - imageR (Django static object): SVG markup for image of radical species
                                     (i.e. A• or B•), obtained by calling
                                     make_images
    - imageH (Django static object): SVG markup for image of hydrognated species
                                     (i.e. A-H or B-H), obtained by calling
                                     make_images
    """

    def __init__(self, smilesR, smilesH, hcount):
        self.smilesR = smilesR
        self.smilesH = smilesH
        self.hcount = hcount

        # Assign self.atomR and self.atomH
        self._assign_atomR()
        self._assign_atomH()

        self.DGdd = None
        self.select = None

        self.imageR = None
        self.imageH = None

        print(self.smilesR, self.atomR, self.smilesH, self.atomH)

    def _assign_atomR(self):
        """
        Assigns self.atomR property by finding which atom number contains
        the radical (in RDKit Mol object corresponding to self.smilesR)
        """
        molR = Chem.MolFromSmiles(self.smilesR)
        for atom in molR.GetAtoms():
            if atom.GetNumRadicalElectrons() == 1:
                self.atomR = atom.GetIdx()
                break

    def _assign_atomH(self):
        """
        Assigns self.atomH property by finding which atom number in the
        hydrogenated species (in RDKit Mol object corresponding to self.smilesH)
        loses an H atom to form the radical 
        """
        molH = Chem.MolFromSmiles(self.smilesH)
        for atom in molH.GetAtoms():
            hcount = atom.GetTotalNumHs()
            rcount = atom.GetNumRadicalElectrons()

            if hcount > 0:
                atom.SetNumRadicalElectrons(rcount + 1)
                atom.SetNumExplicitHs(hcount - 1)

                if Chem.MolToSmiles(molH) == self.smilesR:
                    self.atomH = atom.GetIdx()
                    break

                atom.SetNumRadicalElectrons(rcount)
                atom.SetNumExplicitHs(hcount)


def anion_to_radical_dict(smiles_anion):
    """
    Converts SMILES string of an anion into a dictionary containing
    the corresponding radical

    Arguments:
        smiles_anion (str): SMILES string of the anion

    Returns:
        radical_dict (dict): dictionary with a single entry
            whose key is the SMILES string of the corresponding radical
            and whose value is a Radical object representing that radical

    If smiles_anion does not contain exactly one anion, the function returns
    None.
    """

    radical_dict = None

    mol = Chem.MolFromSmiles(smiles_anion)

    done = False
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() == -1:
            if done:
                # More than one anion
                radical_dict = None
            else:
                hcount = atom.GetTotalNumHs()
                rcount = atom.GetNumRadicalElectrons()

                atom.SetFormalCharge(0)
                atom.SetNumRadicalElectrons(rcount + 1)
                # Smiles string of corresponding radical
                smilesR = Chem.MolToSmiles(mol)

                atom.SetNumRadicalElectrons(rcount)
                atom.SetNumExplicitHs(hcount + 1)
                # Smiles string of corresponding hydrogenated species
                smilesH = Chem.MolToSmiles(mol)

                radical_dict = {
                    smilesR: Radical(smilesR, smilesH, hcount + 1)
                }
                done = True

    return radical_dict


def neutral_to_radical_dict(smilesH):
    """
    Converts SMILES string of a molecule into a dictionary containing
    all possible radicals formed by removing H atoms from a molecule.

    Arguments:
        mol (RDKit Mol object): the input molecule to enumerate radicals for

    Returns:
        radicals (dict): a dictionary whose keys are SMILES strings of the
                         possible radicals, and whose values are Radical objects
                         representing the radicals
    """

    radical_dict = {}

    smilesH = Chem.CanonSmiles(smilesH)
    mol = Chem.MolFromSmiles(smilesH)

    for atom in mol.GetAtoms():
        hcount = atom.GetTotalNumHs()
        rcount = atom.GetNumRadicalElectrons()
        # Can have a radical on this atom
        if hcount > 0:
            atom.SetNumExplicitHs(hcount - 1)
            atom.SetNumRadicalElectrons(rcount + 1)
            # SMILES string of radical
            smilesR = Chem.MolToSmiles(mol)
            atom.SetNumExplicitHs(hcount)
            atom.SetNumRadicalElectrons(rcount)

            # Already have this radical, so just add to its hcount
            if smilesR in radical_dict:
                radical_dict[smilesR].hcount += hcount
            # Add new radical to dictionary
            else:
                radical_dict[smilesR] = Radical(smilesR, smilesH, hcount)
                
    return radical_dict


def compute_barriers(dict_A, dict_B):
    """
    Predicts free energy barriers for a set of HAT reactions of the form:
    A•   +  B-H   -->   A-H  +   B•
    
    Arguments:
        dict_A (dict): dictionary containing a set of Radical objects for A•
                       For now this function assumes that dict_A contains only
                       a single Radical.
        dict_B (dict): dictionary containing a set of Radical objects for B•

    Returns:
        Nothing, but updates the DGdd attribute of each Radical object in dict_B
        with the free energy barrier (in kcal/mol) of the corresponding HAT
        reaction
    """

    n_rxns = len(dict_B)

    # Create a dictionary containing the SMILES strings and reacting atom numbers
    # in each species for each reaction
    smi_reaction = {
        'DGdd_true': ['0'] * n_rxns,
        'AR_smiles': [r.smilesR for r in dict_A.values()] * n_rxns,
        'AR_atom':   [r.atomR for r in dict_A.values()] * n_rxns,
        'AH_smiles': [r.smilesH for r in dict_A.values()] * n_rxns,
        'AH_atom':   [r.atomH for r in dict_A.values()] * n_rxns,
        'BR_smiles': [r.smilesR for r in dict_B.values()],
        'BR_atom':   [r.atomR for r in dict_B.values()],
        'BH_smiles': [r.smilesH for r in dict_B.values()],
        'BH_atom':   [r.atomH for r in dict_B.values()],
    }

    # Django root directory containing manage.py
    project_root = settings.BASE_DIR

    smiles_path = os.path.join(project_root, 'model/inference/smiles.csv')
    descriptors_path = os.path.join(project_root, 'model/inference/descriptors.csv')
    barriers_path = os.path.join(project_root, 'model/inference/barriers.csv')
    scratch_path = os.path.join(project_root, 'model/inference')
    model_path = os.path.join(project_root, 'model/xgb_model.json')

    # Write reactions to smiles.csv file
    df=pd.DataFrame(smi_reaction)
    df.to_csv(smiles_path,index_label='reaction')

    # Compute and output reaction descriptors to descriptors.csv file
    write_reaction_descriptors(in_file = smiles_path,
                               out_file = descriptors_path,
                               scratch_dir = scratch_path)
        
    # Predict reaction barriers and output to barriers.csv file
    write_reaction_barriers(in_file = descriptors_path,
                            out_file = barriers_path,
                            model_file = model_path)
    
    # Read predicted reaction barriers and return as list
    df=pd.read_csv(barriers_path)
    DGdd_list=list(df.loc[:, 'DGdd_pred'])
    
    # Update DGdd attribute of each Radical object in dict_B
    for r, DGdd in zip(dict_B.values(), DGdd_list):
        r.DGdd = DGdd


def compute_selectivities(dict_B):
    """
    Computes selectivities (in %) at room temperature (298 K) for a set of HAT
    reactions of the form:
    A•   +  B-H   -->   A-H  +   B•
    with different radicals B•

    Arguments:
        dict_B (dict): dictionary containing a set of Radical objects for B•
                       These radicals must already have a DGdd attribute, as by
                       calling compute_barriers

    Returns:
        Nothing, but updates the select attribute of each Radical object in dict_B
        with the selectivity (in %) of the corresponding HAT reaction
    """

    R = 8.314
    T = 298
    J_to_kcal = 1 / 4184

    free_energies = np.array([r.DGdd for r in dict_B.values()])
    selectivities = (free_energies - free_energies.min()) / (R * T * J_to_kcal)
    selectivities = np.exp(-selectivities) * np.array([r.hcount for r in dict_B.values()])
    selectivities = 100 * selectivities / selectivities.sum()

    # Update select attribute of each Radical object in dict_B
    for r, s in zip(dict_B.values(), selectivities):
        r.select = s


def make_images(dict_A, dict_B):
    """
    Generates images of all species in a set of HAT reactions of the form:
    A•   +  B-H   -->   A-H  +   B•
    with different radicals B•

    The images are saved in the 'static/images' directory.

    Arguments:
        dict_A (dict): dictionary containing a set of Radical objects for A•
                       For now this function assumes that dict_A contains only
                       a single Radical.
        dict_B (dict): dictionary containing a set of Radical objects for B•

    Returns:
        Nothing, but updates the imageR and imageH attributes of each Radical
        object with SVG markup of the appropriate images
    """

    # Get A• and A-H as RDKit Mol objects
    rA = list(dict_A.values())[0]
    mol_AH = Chem.MolFromSmiles(rA.smilesH)
    mol_AR = Chem.MolFromSmiles(rA.smilesR)

    # Align A• to A-H
    AllChem.Compute2DCoords(mol_AH)
    AllChem.GenerateDepictionMatching2DStructure(mol_AR, mol_AH)

    # Generate and store SVG markup for image of A-H
    d = rdMolDraw2D.MolDraw2DSVG(250, 250)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol_AH)
    rA.imageH = d.GetDrawingText()

    # Generate and store SVG markup for image of A• with highlighted radical
    d = rdMolDraw2D.MolDraw2DSVG(250, 250)
    d.drawOptions().setHighlightColour((0.5, 1.0, 1.0, 0.8))
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol_AR, highlightAtoms=[rA.atomR])
    rA.imageR = d.GetDrawingText()

    for rB in dict_B.values():

        # Get B• and B-H as RDKit Mol objects
        mol_BH = Chem.MolFromSmiles(rB.smilesH)
        mol_BR = Chem.MolFromSmiles(rB.smilesR)

        # Align B• to B-H
        AllChem.Compute2DCoords(mol_BH)
        AllChem.GenerateDepictionMatching2DStructure(mol_BR, mol_BH)

        # Generate and store SVG markup for image of B-H
        d = rdMolDraw2D.MolDraw2DSVG(250, 250)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol_BH)
        rB.imageH = d.GetDrawingText()

        # Generate and store SVG markup for image of B• with highlighted radical
        d = rdMolDraw2D.MolDraw2DSVG(250, 250)
        d.drawOptions().setHighlightColour((1.0, 0.5, 0.5, 0.7))
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol_BR, highlightAtoms=[rB.atomR])
        rB.imageR = d.GetDrawingText()


def validate_smiles_A(smiles):
    """
    Validates the user-supplied SMILES string representing the radical A•
    by ensuring that it contains only a single negative charge (anion) and
    that all atoms are supported by HATPredict.

    Arguments:

        smiles (str): user-supplied SMILES string representing the radical A•

    Returns:
    
        string describing error message if the SMILES string is invalid,
        otherwise None
    """

    allowed_atoms = {'C', 'H', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}
    
    if not smiles:
        return f"""
            Please enter a radical for <strong>A•</strong>.  Use a single negative
            charge (anion) where you want the radical to be.
            """
    
    # Parse the SMILES string into an RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"""
            Please enter a valid radical for <strong>A•</strong>.  Use a single negative
            charge (anion) where you want the radical to be.
            """
    
    negative_charge_count = 0
    
    # Iterate through the atoms in the molecule
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        atom_charge = atom.GetFormalCharge()
        
        # Check if the atom is allowed
        if atom_symbol not in allowed_atoms:
            return f"""
                Please revise your input for <strong>A•</strong> so it no longer contains a
                <strong>{atom_symbol}</strong> atom.  HATPredict currently only accepts
                <strong>C</strong>, <strong>H</strong>, <strong>N</strong>, <strong>O</strong>,
                <strong>F</strong>, <strong>P</strong>, <strong>S</strong>, <strong>Cl</strong>,
                <strong>Br</strong>, and <strong>I</strong> atoms.
                """
        
        # Check if the atom has a positive charge
        if atom_charge > 0:
            return f"""
                Please revise the positively-charged <strong>{atom_symbol}</strong> atom
                in your input for <strong>A•</strong>.  HATPredict currently does not accept
                positively-charged atoms.  
                """
        
        # Count atoms with a negative charge
        if atom_charge < 0:
            negative_charge_count += 1
    
    if negative_charge_count != 1:
        return f"""
            Please revise your input for <strong>A•</strong> so it contains exactly
            one negative charge where you want the radical to be.  (HATPredict
            currently does not accept any other negatively-charged atoms.)
            """
    
    return None


def validate_smiles_B(smiles):
    """
    Validates the user-supplied SMILES string representing the molecule B-H
    by ensuring that it contains no charged atoms and that all atoms are
    supported by HATPredict.

    Arguments:

        smiles (str): user-supplied SMILES string representing the molecule B-H

    Returns:

        string describing error message if the SMILES string is invalid,
        otherwise None
    """

    allowed_atoms = {'C', 'H', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}
    
    if not smiles:
        return f"""
            Please enter a molecule for <strong>B–H</strong>.
            """
    
    # Parse the SMILES string into an RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"""
            Please enter a valid molecule for <strong>B–H</strong>.
            """
    
    # Iterate through the atoms in the molecule
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        atom_charge = atom.GetFormalCharge()
        
        # Check if the atom is allowed
        if atom_symbol not in allowed_atoms:
            return f"""
                Please revise your input for <strong>B–H</strong> so it no longer contains a
                <strong>{atom_symbol}</strong> atom.  HATPredict currently only accepts
                <strong>C</strong>, <strong>H</strong>, <strong>N</strong>, <strong>O</strong>,
                <strong>F</strong>, <strong>P</strong>, <strong>S</strong>, <strong>Cl</strong>,
                <strong>Br</strong>, and <strong>I</strong> atoms.
                """
        
        # Check if the atom has a charge
        if atom_charge != 0:
            return f"""
                Please revise the charged <strong>{atom_symbol}</strong> atom in your input
                for <strong>B–H</strong>.  HATPredict currently does not accept charged atoms
                in the molecule <strong>B–H</strong>.
                """
    
    return None


def index(request):
    """
    View function for the index page.

    If both get parameters 'smilesA' or 'smilesB' are missing, this function 
    renders the home page for barrier and selectivity predictions of HAT reactions
    that is ready for user input.

    Otherwise, the function validates the parameters (rendering an error message if
    either parameter is not valid), and then computes the barriers and
    selectivities for a set of HAT reactions of the form:
    A•   +  B-H   -->   A-H  +   B•
    with different radicals B•

    Then the function constructs a dictionary dict_A containing a Radical object 
    for A• and a dictionary dict_B containing Radical objects for all possible 
    radicals B•.  It calls functions to compute barriers and selectivites, and to 
    generate images of all molecules.  Finally, it generates the string html_output
    containing HTML to display the final results to the user.
    
    Arguments:
        request: Django web request
        
    Returns:
        Django render object containing HTML needed to display the results to the user
    """

    smi_A=request.GET.get('smilesA')    
    smi_B=request.GET.get('smilesB')

    # If no parameters are provided, serve the main page ready for user input
    if not smi_A and not smi_B:
        return render(request, 'main.html')
    
    # Validate the smilesA string, and display an error message if it is invalid
    errorA = validate_smiles_A(smi_A)
    if errorA != None:
        html_output = f"""
            <div class="alert alert-warning" role="alert">
                {errorA}
            </div>
        """
        return render(request, 'main.html', {'html_output': html_output})

    # Validate the smilesB string, and display an error message if it is invalid
    errorB = validate_smiles_B(smi_B)
    if errorB != None:
        html_output = f"""
            <div class="alert alert-warning" role="alert">
                {errorB}
            </div>
        """
        return render(request, 'main.html', {'html_output': html_output})

    # Generate dictionary containing a single Radical object for A•
    dict_A = anion_to_radical_dict(smi_A)
    
    # Generate dictionary containing Radical objects for all possible B• radicals
    dict_B = neutral_to_radical_dict(smi_B)

    # Compute barriers and selectivites, and generate images of molecules
    compute_barriers(dict_A, dict_B)
    compute_selectivities(dict_B)
    make_images(dict_A, dict_B)

    rA = list(dict_A.values())[0]
    rB = list(dict_B.values())[0]

    # Generate HTML table of HAT reaction whose selectivity is being computed
    html_output = f"""
        <h3 style="font-weight: normal;">Here is the <strong>reaction</strong> you computed:</h3>
        <table class="mx-auto my-4">
            <tbody>
                <tr>
                    <td>{rA.imageR}</td>
                    <td class="px-3"><h3>+</h3></td>               
                    <td>{rB.imageH}</td>
                    <td class="px-3"><h3>&srarr;</h3></td>
                    <td>{rA.imageH}</td>
                    <td class="px-3"><h3>+</h3></td>
                    <td><h3>Radical Products</h3></td>
                </tr>
            </tbody>
        </table>
    """

    html_output += f"""
        <h3 style="font-weight: normal;">Here are the <strong>radical products</strong>, along with their predicted <strong>free energy barriers (∆G<sup>‡</sup>)</strong> and <strong>selectivities</strong>:</h3>
    """

    # Generate HTML table of all radical products with their barriers and selectivities
    html_output += f"""
        <table class="mx-auto my-4">
            <tbody>
    """
    
    # Loop over radical products B• from highest to lowest selectivity
    for i, rB in enumerate(sorted(dict_B.values(), key=lambda r: r.select, reverse=True)):
        if i % 3 == 0:
            html_output += f"""
                <tr>
            """
        html_output += f"""
            <td class="px-5 py-3">
                <div class="text-center">
                    {rB.imageR}
                </div>
                <div class="text-start">
                    <h3>Barrier:&nbsp;&nbsp;<span style='font-weight: normal;'>{rB.DGdd:.1f} kcal/mol</span></h3>
                    <h3># of Hs:&nbsp;&nbsp;<span style='font-weight: normal;'>{rB.hcount}</span></h3>
                    <h3>Selectivity:&nbsp;&nbsp;<span style='font-weight: normal;'>{rB.select:.0f}%</span></h3>
                </div>
            </td>
        """
        if i % 3 == 2:
            html_output += f"""
                </tr>
            """
    if i % 3 != 2:
        html_output += f"""
            </tr>
        """

    html_output += f"""
            </tbody>
        </table>
    """
                
    return render(request, 'main.html', {'html_output': html_output})


def about(request):
    """
    View function for the about page.

    This function reads the md/about.md static markdown file and converts it to
    HTML, which is then passed to the about.html template.

    Arguments:
        request: Django web request
        
    Returns:
        Django render object containing HTML needed to display about page to user
    """

    # Read about.md static file
    md_path = os.path.join(settings.STATIC_ROOT, "md/about.md")
    md = open(md_path, encoding='utf-8').read()

    # Convert to HTML
    html_output = markdown2.markdown(md)

    return render(request, "about.html", {"html_output": html_output})