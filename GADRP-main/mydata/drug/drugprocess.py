import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdMolDescriptors, Lipinski, Crippen
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np
import re

print(f"RDKit version: {rdkit.__version__}")
print(f"Python version: 3.6 (Assuming you're using a Python 3.6 environment)")


def calculate_descriptors(smiles_string, all_descriptor_names):
    """Calculates molecular descriptors using RDKit."""

    descr_dict = {name: None for name in all_descriptor_names}

    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return descr_dict
    except Exception as e:
        print(f"Error creating molecule from SMILES: {smiles_string} - {e}")
        return descr_dict

    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        print(f"Error generating 3D coordinates for: {smiles_string} - {e}")

    try:
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            [desc[0] for desc in Descriptors._descList if desc[0] in all_descriptor_names and desc[0] not in ['Kier_shape_1','Kier_shape_2','Zagreb_group_index_1','Zagreb_group_index_2']]
        )

        descrs = calculator.CalcDescriptors(mol)
        for name, value in zip(calculator.GetDescriptorNames(), descrs):
            descr_dict[name] = value

    except Exception as e:
        print(f"Error calculating _descList descriptors: {e}")


    # --- INDIVIDUAL TRY-EXCEPT BLOCKS FOR EACH CALCULATION ---

    # Crippen LogP and logP - Corrected section
    try:
        descr_dict['LogP'] = Crippen.MolLogP(mol)  # First, try RDKit's built-in for LogP
    except Exception:
        try:
            descr_dict['LogP'] = calculate_logp(mol) # Then, use the fallback for LogP
        except:
            pass # if they both fail

    try:
        descr_dict['logP'] = calculate_logp(mol) # Calculate logP using the fallback function
    except Exception:
        pass # if it fail, logP remains None
    # ... (rest of the descriptor calculations - qed, Molecular Weight, Lipinski, etc.) ...

    try:
        descr_dict['qed'] = Descriptors.qed(mol)
    except Exception:
        pass

    try:
        descr_dict['Molecular_weight'] = Descriptors.MolWt(mol)
    except Exception:
        pass

    # Lipinski and related
    try:
        descr_dict['Number_of_HBA_1'] = Lipinski.NumHAcceptors(mol)
    except Exception:
        pass

    try:
        descr_dict['Number_of_HBA_2'] = rdMolDescriptors.CalcNumHBA(mol)
    except Exception:
        pass

    try:
        descr_dict['Number_of_HBD_1'] = Lipinski.NumHDonors(mol)
    except Exception:
        pass

    try:
        descr_dict['Number_of_HBD_2'] = rdMolDescriptors.CalcNumHBD(mol)
    except Exception:
        pass

    try:
        descr_dict['Number_of_acidic_groups'] = Lipinski.NumHDonors(mol)  # Placeholder
    except Exception:
        pass

    try:
        descr_dict['Number_of_aliphatic_OH_groups'] = calculate_num_aliphatic_oh_groups(mol) # Manual
    except Exception:
        pass

    try:
        descr_dict['Number_of_basic_groups'] = Lipinski.NumHAcceptors(mol)  # Placeholder
    except Exception:
        pass

    try:
        descr_dict['Fraction_of_rotatable_bonds'] = calculate_fraction_rotatable_bonds(mol) # Manual
    except Exception:
        pass

    try:
        descr_dict['Number_of_heavy_bonds'] = mol.GetNumBonds() - mol.GetNumAtoms(onlyExplicit=False) + mol.GetNumBonds(onlyHeavy=True)
    except Exception:
        pass
    try:
        descr_dict['Number_of_heterocycles'] =  calculate_num_heterocycles(mol) #Manual
    except Exception:
        pass
    try:
        descr_dict['Number_of_hydrophobic_groups'] =  calculate_num_hydrophobic_groups(mol) # Manual
    except Exception:
      pass

    try:
        descr_dict['MolarRefractivity'] = Crippen.MolMR(mol)
    except Exception:
        pass

    try:
        descr_dict['Number_of_bonds'] = mol.GetNumBonds()
    except Exception:
        pass

    try:
        descr_dict['Number_of_NO2_groups'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[N+](=O)[O-]')))
    except Exception:
        pass

    try:
        descr_dict['Number_of_SO_groups'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('S(=O)')))
    except Exception:
        pass

    try:
        descr_dict['Number_of_OSO_groups'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('OS(=O)(=O)O')))
    except Exception:
        pass

    try:
        descr_dict['Number_of_SO2_groups'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('S(=O)(=O)')))
    except Exception:
        pass
    try:
        descr_dict['PolarSurfaceArea'] = Descriptors.TPSA(mol)
    except Exception:
        pass

    # --- 3D Descriptors (INDIVIDUAL TRY-EXCEPT) ---
    try:
        descr_dict['Geometrical_diameter'] = Descriptors3D.RadiusOfGyration(mol) if Descriptors3D.RadiusOfGyration(mol) != float('inf') else None
    except Exception:
        pass

    try:
        descr_dict['Geometrical_radius'] = Descriptors3D.InertialShapeFactor(mol) if Descriptors3D.InertialShapeFactor(mol) != float('inf') else None
    except Exception:
        pass

    try:
        descr_dict['Geometrical_shape_coefficient'] = Descriptors3D.Eccentricity(mol) if Descriptors3D.Eccentricity(mol) != float('inf') else None
    except Exception:
        pass

    #Kier and zagreb calculations
    try:
        descr_dict['Kier_shape_1'] = calc_kier_shape_1(mol)
    except Exception:
        pass
    try:
        descr_dict['Kier_shape_2'] = calc_kier_shape_2(mol)
    except Exception:
      pass
    try:
        descr_dict['Zagreb_group_index_1'] = calc_zagreb_group_index_1(mol)
    except Exception:
        pass
    try:
        descr_dict['Zagreb_group_index_2'] = calc_zagreb_group_index_2(mol)
    except Exception:
        pass
    try:
      descr_dict['Ncharges'] = calculate_ncharges(mol)
    except Exception:
      pass



    # Atom and bond counts
    try:
        descr_dict['atoms'] = mol.GetNumAtoms()
    except Exception:
        pass

    try:
        descr_dict['abonds'] = calculate_num_aromatic_bonds(mol) # Manual
    except Exception:
        pass

    try:
        descr_dict['dbonds'] = calculate_num_double_bonds(mol) # Manual
    except Exception:
        pass

    try:
        descr_dict['sbonds'] = calculate_num_single_bonds(mol) # Manual
    except Exception:
        pass

    try:
        descr_dict['tbonds'] = calculate_num_triple_bonds(mol) # Manual
    except Exception:
        pass
    try:
      descr_dict['C'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    except Exception:
      pass
    try:
       descr_dict['H'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'H')
    except Exception:
      pass
    try:
       descr_dict['N'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
    except Exception:
        pass
    try:
      descr_dict['O'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
    except Exception:
        pass
    try:
      descr_dict['F'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    except Exception:
       pass
    try:
      descr_dict['Cl'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl')
    except Exception:
       pass
    try:
      descr_dict['Br'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br')
    except Exception:
       pass
    try:
      descr_dict['S'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
    except Exception:
        pass
    try:
       descr_dict['P'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'P')
    except Exception:
      pass
    try:
     descr_dict['B'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'B')
    except Exception:
       pass
    try:
      descr_dict['I'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'I')
    except Exception:
       pass
    try:
     descr_dict['K'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'K')
    except Exception:
      pass
    try:
      descr_dict['Sb'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Sb')
    except Exception:
        pass
    try:
      descr_dict['Hg'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Hg')
    except Exception:
      pass
    try:
       descr_dict['Pt'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Pt')
    except Exception:
        pass
    try:
      descr_dict['V'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'V')
    except Exception:
        pass
    try:
      descr_dict['Zn'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Zn')
    except Exception:
       pass
    try:
      descr_dict['Cr'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cr')
    except Exception:
      pass
    try:
      descr_dict['Se'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Se')
    except Exception:
        pass
    try:
       descr_dict['As'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'As')
    except Exception:
      pass
    try:
       descr_dict['nF'] = descr_dict['F']
    except Exception:
      pass

    # SMARTS-based counts
    try:
        descr_dict['RNH2'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH2][CX4]')))
    except Exception:
        pass

    try:
        descr_dict['R2NH'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH1]([CX4])[CX4]')))
    except Exception:
        pass

    try:
        descr_dict['R3N'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3]([CX4])([CX4])[CX4]')))
    except Exception:
        pass

    try:
        descr_dict['ROPO3'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OX2][PX4](=[OX1])([OX2])[OX2]')))
    except Exception:
        pass

    try:
        descr_dict['ROH'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OX2H][CX4]')))
    except Exception:
        pass
    try:
      descr_dict['RCHO'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3H1](=O)[#6]')))
    except Exception:
      pass
    try:
      descr_dict['RCOR'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=[OX1])([#6])[#6]')))
    except Exception:
      pass
    try:
      descr_dict['RCOOH'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[OX2H1]')))
    except Exception:
      pass
    try:
      descr_dict['RCOOR'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[OX2H0][#6]')))
    except Exception:
      pass
    try:
      descr_dict['ROR'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OX2H0]([#6])[#6]')))
    except Exception:
      pass
    try:
       descr_dict['RCCH'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C]#[C]')))
    except Exception:
       pass
    try:
      descr_dict['RCN'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NX1]#[CX2]')))
    except Exception:
        pass
    try:
      descr_dict['RINGS'] = rdMolDescriptors.CalcNumRings(mol)
    except Exception:
        pass
    try:
       descr_dict['AROMATIC'] = rdMolDescriptors.CalcNumAromaticRings(mol)
    except Exception:
      pass

    return descr_dict

def calculate_ncharges(mol):
    """Calculates the net charge of a molecule (sum of formal charges)."""
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

def calc_kier_shape_1(mol):
    """Calculates Kier Shape Index 1."""
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    num_path_order_2 = len(Chem.FindAllPathsOfLengthN(mol, 2))
    alpha = sum([(atom.GetAtomicNum() - atom.GetTotalValence() + atom.GetTotalNumHs()) / (atom.GetAtomicNum() - atom.GetTotalNumHs() -1) for atom in mol.GetAtoms() if (atom.GetAtomicNum() - atom.GetTotalNumHs() -1) != 0])
    return (num_atoms + alpha -1 ) * (num_atoms + alpha -1) / (num_path_order_2 + alpha)

def calc_kier_shape_2(mol):
    """Calculates Kier Shape Index 2."""
    num_atoms = mol.GetNumAtoms()
    num_path_order_2 = len(Chem.FindAllPathsOfLengthN(mol, 2))
    num_path_order_3 = len(Chem.FindAllPathsOfLengthN(mol, 3))
    alpha = sum([(atom.GetAtomicNum() - atom.GetTotalValence() + atom.GetTotalNumHs()) / (atom.GetAtomicNum() - atom.GetTotalNumHs() -1) for atom in mol.GetAtoms() if (atom.GetAtomicNum() - atom.GetTotalNumHs() -1) != 0])
    return (num_atoms + alpha - 2) * (num_atoms + alpha - 2) / (num_path_order_3 + alpha)

def calc_zagreb_group_index_1(mol):
    """Calculates Zagreb Group Index 1."""
    return sum(atom.GetDegree()**2 for atom in mol.GetAtoms())
def calc_zagreb_group_index_2(mol):
    """Calculate Zagreb Group Index 2"""
    zagreb2 = 0
    for bond in mol.GetBonds():
      zagreb2 += bond.GetBeginAtom().GetDegree() * bond.GetEndAtom().GetDegree()
    return zagreb2

def calculate_num_aliphatic_oh_groups(mol):
    """Calculates the number of aliphatic OH groups using SMARTS."""
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OX2H][CX4]')))

def calculate_fraction_rotatable_bonds(mol):
    """Calculates the fraction of rotatable bonds."""
    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_bonds = mol.GetNumBonds()
    return float(num_rotatable_bonds) / num_bonds if num_bonds > 0 else 0.0

def calculate_num_heterocycles(mol):
    """Calculates the number of heterocycles using SMARTS."""
    ri = mol.GetRingInfo()
    return ri.NumRings() - len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]1[#6][#6][#6][#6][#6]1'))) #  substract the aromatic rings

def calculate_num_double_bonds(mol):
  """Calculate the number of double bonds"""
  num = 0
  for bond in mol.GetBonds():
    if bond.GetBondType() == Chem.BondType.DOUBLE:
      num +=1
  return num

def calculate_num_single_bonds(mol):
  """Calculate the number of single bonds"""
  num = 0
  for bond in mol.GetBonds():
    if bond.GetBondType() == Chem.BondType.SINGLE:
      num +=1
  return num

def calculate_num_triple_bonds(mol):
  """Calculate the number of single bonds"""
  num = 0
  for bond in mol.GetBonds():
    if bond.GetBondType() == Chem.BondType.TRIPLE:
      num +=1
  return num

def calculate_logp(mol):
    """
    Calculate the LogP using Crippen's method.
    This is a fallback in case Crippen.MolLogP fails.  It's *less*
    accurate than the built-in Crippen.MolLogP, but it's better than
    nothing, and it avoids the need for an external dependency.
    This is a *very* simplified implementation.  For real-world use,
    you would need a much more sophisticated approach with more
    atom types and corrections.
    """
    # Simplified atom contributions (VERY approximate)
    atom_contribs = {
        'C': 0.2,
        'H': 0.01,
        'O': -0.4,
        'N': -0.3,
        'S': 0.1,
        'Cl': 0.4,
        'Br': 0.6,
        'I': 0.8,
        'F': -0.2,
        # Add more atom types as needed...
    }
    logp = 0.0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atom_contribs:
            logp += atom_contribs[symbol]
        #else: # Consider throwing a warning for unknown atoms
        #    print(f"Warning: Unrecognized atom type: {symbol}")
    return logp

def calculate_num_aromatic_bonds(mol):
    """Calculates the number of aromatic bonds in a molecule."""
    return sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())

def calculate_num_hydrophobic_groups(mol):
    """
    Calculates the number of hydrophobic groups in a molecule.  This is a
    simplified implementation using SMARTS for common hydrophobic groups.
    For a more robust calculation, consider using a more comprehensive set
    of SMARTS patterns or a different method entirely.
    """
    # Define SMARTS patterns for common hydrophobic groups
    hydrophobic_smarts = [
        '[CX4;CH3]',   # Methyl
        '[CX4;CH2]',   # Methylene
        '[CX4;CH]',  # Methine
        '[CX4;C]([CX4])([CX4])([CX4])',   # Quaternary carbon
        '[F,Cl,Br,I]~[#6]', # Halogen connected to carbon
        'c1ccccc1',       # Phenyl
        'c1ncccc1', # Pyridine like
        'c1ncc(=O)[nH]c1',  # Pyrimidones
        'c1nn[nH]n1',    # imidazole like
    ]
    count = 0
    for pattern in hydrophobic_smarts:
      count += len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
    return count

def calculate_descriptors_from_file(input_file, output_file, all_descriptor_names):
    """
    Reads SMILES from a CSV file, calculates descriptors, and writes the results to another CSV file.

    Args:
        input_file (str): Path to the input CSV file (drug_name_cid.csv).
        output_file (str): Path to the output CSV file (269dim.csv).
        all_descriptor_names (list): List of all descriptor names.
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Check if required columns exist
    required_columns = ['CID', 'CanonicalSMILES', 'IsomericSMILES']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Input file must contain columns: {required_columns}")
        return

    # Use Canonical SMILES if available, otherwise use Isomeric SMILES.  Prioritize Canonical.
    df['SMILES'] = np.where(df['CanonicalSMILES'].notna() & (df['CanonicalSMILES'] != ''), df['CanonicalSMILES'], df['IsomericSMILES'])

    # Calculate descriptors
    results = []
    for index, row in df.iterrows():
        smiles = row['SMILES']
        cid = row['CID']  # Get CID
        descriptors = calculate_descriptors(smiles, all_descriptor_names)
        descriptors['CID'] = cid  # Add CID
        results.append(descriptors)

    result_df = pd.DataFrame(results)

    # Reorder columns to put 'CID' at the beginning
    if 'CID' in result_df.columns:
        cols = ['CID'] + [col for col in result_df.columns if col != 'CID']
        result_df = result_df[cols]
    else:
        print("Warning: 'CID' column not found in the results.")

    # Save results
    try:
        result_df.to_csv(output_file, index=False)
        print(f"Descriptors successfully saved to '{output_file}'")
    except Exception as e:
        print(f"Error writing output file: {e}")


# --- Example Usage ---
if __name__ == '__main__':
    input_csv_file = 'drug.csv'  # Replace with your input file
    output_csv_file = '269dim.csv'  # Replace with your desired output file

    # --- ALL DESCRIPTOR NAMES (YOUR LIST) ---
    all_descriptors = ['pubchem_cid', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed', 'Molecular_weight', 'LogP', 'Number_of_HBA_1', 'Number_of_HBA_2', 'Number_of_HBD_1', 'Number_of_HBD_2', 'Number_of_acidic_groups', 'Number_of_aliphatic_OH_groups', 'Number_of_basic_groups', 'Fraction_of_rotatable_bonds', 'Number_of_heavy_bonds', 'Number_of_heterocycles', 'Number_of_hydrophobic_groups', 'MolarRefractivity', 'Number_of_bonds', 'Number_of_NO2_groups', 'Number_of_SO_groups', 'Number_of_OSO_groups', 'Number_of_SO2_groups', 'PolarSurfaceArea', 'Geometrical_diameter', 'Geometrical_radius', 'Geometrical_shape_coefficient', 'Kier_shape_1', 'Kier_shape_2', 'Zagreb_group_index_1', 'Zagreb_group_index_2', 'Ncharges', 'C', 'H', 'N', 'O', 'RNH2', 'R2NH', 'R3N', 'ROPO3', 'ROH', 'RCHO', 'RCOR', 'RCOOH', 'RCOOR', 'ROR', 'RCCH', 'RCN', 'RINGS', 'AROMATIC', 'F', 'Cl', 'Br', 'S', 'P', 'B', 'I', 'K', 'Sb', 'Hg', 'Pt', 'V', 'Zn', 'Cr', 'Se', 'As', 'abonds', 'atoms', 'dbonds', 'logP', 'nF', 'sbonds', 'tbonds']
    calculate_descriptors_from_file(input_csv_file, output_csv_file, all_descriptors)