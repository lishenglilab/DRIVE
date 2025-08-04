import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm  # 导入tqdm库
import os
GLOBAL_CONDITIONS = {
        '>=_4_H': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'H']) >= 4,
        '>=_8_H': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'H']) >= 8,
        '>=_16_H': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'H']) >= 16,
        '>=_32_H': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'H']) >= 32,
        '>=_1_Li': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Li']) >= 1,
        '>=_2_Li': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Li']) >= 2,
        '>=_1_B': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'B']) >= 1,
        '>=_2_B': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'B']) >= 2,
        '>=_4_B': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'B']) >= 4,
        '>=_2_C': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'C']) >= 2,
        '>=_4_C': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'C']) >= 4,
        '>=_8_C': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'C']) >= 8,
        '>=_16_C': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'C']) >= 16,
        '>=_32_C': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'C']) >= 32,
        '>=_1_N': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'N']) >= 1,
        '>=_2_N': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'N']) >= 2,
        '>=_4_N': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'N']) >= 4,
        '>=_8_N': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'N']) >= 8,
        '>=_1_O': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'O']) >= 1,
        '>=_2_O': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'O']) >= 2,
        '>=_4_O': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'O']) >= 4,
        '>=_8_O': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'O']) >= 8,
        '>=_16_O': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'O']) >= 16,
        '>=_1_F': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'F']) >= 1,
        '>=_2_F': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'F']) >= 2,
        '>=_4_F': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'F']) >= 4,
        '>=_1_Na': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Na']) >= 1,
        '>=_2_Na': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Na']) >= 2,
        '>=_1_Si': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Si']) >= 1,
        '>=_2_Si': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Si']) >= 2,
        '>=_1_P': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'P']) >= 1,
        '>=_2_P': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'P']) >= 2,
        '>=_4_P': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'P']) >= 4,
        '>=_1_S': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'S']) >= 1,
        '>=_2_S': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'S']) >= 2,
        '>=_4_S': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'S']) >= 4,
        '>=_8_S': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'S']) >= 8,
        '>=_1_Cl': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Cl']) >= 1,
        '>=_2_Cl': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Cl']) >= 2,
        '>=_4_Cl': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Cl']) >= 4,
        '>=_8_Cl': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Cl']) >= 8,
        '>=_1_K': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'K']) >= 1,
        '>=_2_K': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'K']) >= 2,
        '>=_1_Br': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Br']) >= 1,
        '>=_2_Br': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Br']) >= 2,
        '>=_4_Br': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Br']) >= 4,
        '>=_1_I': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'I']) >= 1,
        '>=_2_I': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'I']) >= 2,
        '>=_4_I': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'I']) >= 4,
        '>=_1_Be': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Be']) >= 1,
        '>=_1_Mg': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Mg']) >= 1,
        '>=_1_Al': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Al']) >= 1,
        '>=_1_Ca': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ca']) >= 1,
        '>=_1_Sc': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Sc']) >= 1,
        '>=_1_Ti': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ti']) >= 1,
        '>=_1_V': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'V']) >= 1,
        '>=_1_Cr': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Cr']) >= 1,
        '>=_1_Mn': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Mn']) >= 1,
        '>=_1_Fe': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Fe']) >= 1,
        '>=_1_Co': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Co']) >= 1,
        '>=_1_Ni': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ni']) >= 1,
        '>=_1_Cu': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Cu']) >= 1,
        '>=_1_Zn': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Zn']) >= 1,
        '>=_1_Ga': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ga']) >= 1,
        '>=_1_Ge': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ge']) >= 1,
        '>=_1_As': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'As']) >= 1,
        '>=_1_Se': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Se']) >= 1,
        '>=_1_Kr': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Kr']) >= 1,
        '>=_1_Rb': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Rb']) >= 1,
        '>=_1_Sr': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Sr']) >= 1,
        '>=_1_Y': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Y']) >= 1,
        '>=_1_Zr': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Zr']) >= 1,
        '>=_1_Nb': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Nb']) >= 1,
        '>=_1_Mo': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Mo']) >= 1,
        '>=_1_Ru': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ru']) >= 1,
        '>=_1_Rh': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Rh']) >= 1,
        '>=_1_Pd': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Pd']) >= 1,
        '>=_1_Ag': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ag']) >= 1,
        '>=_1_Cd': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Cd']) >= 1,
        '>=_1_In': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'In']) >= 1,
        '>=_1_Sn': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Sn']) >= 1,
        '>=_1_Sb': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Sb']) >= 1,
        '>=_1_Te': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Te']) >= 1,
        '>=_1_Xe': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Xe']) >= 1,
        '>=_1_Cs': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Cs']) >= 1,
        '>=_1_Ba': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ba']) >= 1,
        '>=_1_Lu': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Lu']) >= 1,
        '>=_1_Hf': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Hf']) >= 1,
        '>=_1_Ta': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ta']) >= 1,
        '>=_1_W': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'W']) >= 1,
        '>=_1_Re': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Re']) >= 1,
        '>=_1_Os': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Os']) >= 1,
        '>=_1_Ir': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ir']) >= 1,
        '>=_1_Pt': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Pt']) >= 1,
        '>=_1_Au': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Au']) >= 1,
        '>=_1_Hg': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Hg']) >= 1,
        '>=_1_Tl': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Tl']) >= 1,
        '>=_1_Pb': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Pb']) >= 1,
        '>=_1_Bi': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Bi']) >= 1,
        '>=_1_La': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'La']) >= 1,
        '>=_1_Ce': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ce']) >= 1,
        '>=_1_Pr': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Pr']) >= 1,
        '>=_1_Nd': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Nd']) >= 1,
        '>=_1_Pm': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Pm']) >= 1,
        '>=_1_Sm': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Sm']) >= 1,
        '>=_1_Eu': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Eu']) >= 1,
        '>=_1_Gd': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Gd']) >= 1,
        '>=_1_Tb': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Tb']) >= 1,
        '>=_1_Dy': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Dy']) >= 1,
        '>=_1_Ho': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Ho']) >= 1,
        '>=_1_Er': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Er']) >= 1,
        '>=_1_Tm': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Tm']) >= 1,
        '>=_1_Yb': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Yb']) >= 1,
        '>=_1_Tc': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'Tc']) >= 1,
        '>=_1_U': lambda x: len([a for a in x.GetAtoms() if a.GetSymbol() == 'U']) >= 1,
        '>=_1_any_ring_size_3': lambda x: any(len(ring) == 3 for ring in x.GetRingInfo().AtomRings()),
        '>=_1_saturated_or_aromatic_carbon-only_ring_size_3': lambda x: any(
            len(ring) == 3 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_nitrogen-containing_ring_size_3': lambda x: any(
            len(ring) == 3 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_heteroatom-containing_ring_size_3': lambda x: any(
            len(ring) == 3 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_carbon-only_ring_size_3': lambda x: any(
            len(ring) == 3 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_nitrogen-containing_ring_size_3': lambda x: any(
            len(ring) == 3 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_heteroatom-containing_ring_size_3': lambda x: any(
            len(ring) == 3 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_2_any_ring_size_3': lambda x: sum(len(ring) == 3 for ring in x.GetRingInfo().AtomRings()) >= 2,
        '>=_2_saturated_or_aromatic_carbon-only_ring_size_3': lambda x: sum(
            len(ring) == 3 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_nitrogen-containing_ring_size_3': lambda x: sum(
            len(ring) == 3 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_heteroatom-containing_ring_size_3': lambda x: sum(
            len(ring) == 3 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_carbon-only_ring_size_3': lambda x: sum(
            len(ring) == 3 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_nitrogen-containing_ring_size_3': lambda x: sum(
            len(ring) == 3 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_heteroatom-containing_ring_size_3': lambda x: sum(
            len(ring) == 3 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_1_any_ring_size_4': lambda x: any(len(ring) == 4 for ring in x.GetRingInfo().AtomRings()),
        '>=_1_saturated_or_aromatic_carbon-only_ring_size_4': lambda x: any(
            len(ring) == 4 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_nitrogen-containing_ring_size_4': lambda x: any(
            len(ring) == 4 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_heteroatom-containing_ring_size_4': lambda x: any(
            len(ring) == 4 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_carbon-only_ring_size_4': lambda x: any(
            len(ring) == 4 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_nitrogen-containing_ring_size_4': lambda x: any(
            len(ring) == 4 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_heteroatom-containing_ring_size_4': lambda x: any(
            len(ring) == 4 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_2_any_ring_size_4': lambda x: sum(len(ring) == 4 for ring in x.GetRingInfo().AtomRings()) >= 2,
        '>=_2_saturated_or_aromatic_carbon-only_ring_size_4': lambda x: sum(
            len(ring) == 4 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_nitrogen-containing_ring_size_4': lambda x: sum(
            len(ring) == 4 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_heteroatom-containing_ring_size_4': lambda x: sum(
            len(ring) == 4 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_carbon-only_ring_size_4': lambda x: sum(
            len(ring) == 4 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_nitrogen-containing_ring_size_4': lambda x: sum(
            len(ring) == 4 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_heteroatom-containing_ring_size_4': lambda x: sum(
            len(ring) == 4 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_1_any_ring_size_5': lambda x: any(len(ring) == 5 for ring in x.GetRingInfo().AtomRings()),
        '>=_1_saturated_or_aromatic_carbon-only_ring_size_5': lambda x: any(
            len(ring) == 5 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_nitrogen-containing_ring_size_5': lambda x: any(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_heteroatom-containing_ring_size_5': lambda x: any(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_carbon-only_ring_size_5': lambda x: any(
            len(ring) == 5 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_nitrogen-containing_ring_size_5': lambda x: any(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_heteroatom-containing_ring_size_5': lambda x: any(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_2_any_ring_size_5': lambda x: sum(len(ring) == 5 for ring in x.GetRingInfo().AtomRings()) >= 2,
        '>=_2_saturated_or_aromatic_carbon-only_ring_size_5': lambda x: sum(
            len(ring) == 5 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_nitrogen-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_heteroatom-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_carbon-only_ring_size_5': lambda x: sum(
            len(ring) == 5 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_nitrogen-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_heteroatom-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_3_any_ring_size_5': lambda x: any(len(ring) == 5 for ring in  x.GetRingInfo().AtomRings()) >= 3,
        '>=_3_saturated_or_aromatic_carbon-only_ring_size_5': lambda x: any(
            len(ring) == 5 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_3_saturated_or_aromatic_nitrogen-containing_ring_size_5': lambda x: any(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_3_saturated_or_aromatic_heteroatom-containing_ring_size_5': lambda x: any(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_3_unsaturated_non-aromatic_carbon-only_ring_size_5': lambda x: any(
            len(ring) == 5 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_3_unsaturated_non-aromatic_nitrogen-containing_ring_size_5': lambda x: any(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_3_unsaturated_non-aromatic_heteroatom-containing_ring_size_5': lambda x: any(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_4_any_ring_size_5': lambda x: sum(len(ring) == 5 for ring in x.GetRingInfo().AtomRings()) >= 4,
        '>=_4_saturated_or_aromatic_carbon-only_ring_size_5': lambda x: sum(
            len(ring) == 5 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_4_saturated_or_aromatic_nitrogen-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_4_saturated_or_aromatic_heteroatom-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_4_unsaturated_non-aromatic_carbon-only_ring_size_5': lambda x: sum(
            len(ring) == 5 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_4_unsaturated_non-aromatic_nitrogen-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_4_unsaturated_non-aromatic_heteroatom-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_5_any_ring_size_5': lambda x: sum(len(ring) == 5 for ring in x.GetRingInfo().AtomRings()) >= 5,
        '>=_5_saturated_or_aromatic_carbon-only_ring_size_5': lambda x: sum(
            len(ring) == 5 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_5_saturated_or_aromatic_nitrogen-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_5_saturated_or_aromatic_heteroatom-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_5_unsaturated_non-aromatic_carbon-only_ring_size_5': lambda x: sum(
            len(ring) == 5 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_5_unsaturated_non-aromatic_nitrogen-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_5_unsaturated_non-aromatic_heteroatom-containing_ring_size_5': lambda x: sum(
            len(ring) == 5 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_1_any_ring_size_6': lambda x: any(len(ring) == 6 for ring in x.GetRingInfo().AtomRings()),
        '>=_1_saturated_or_aromatic_carbon-only_ring_size_6': lambda x: any(
            len(ring) == 6 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_nitrogen-containing_ring_size_6': lambda x: any(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_heteroatom-containing_ring_size_6': lambda x: any(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_carbon-only_ring_size_6': lambda x: any(
            len(ring) == 6 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_nitrogen-containing_ring_size_6': lambda x: any(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_heteroatom-containing_ring_size_6': lambda x: any(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_2_any_ring_size_6': lambda x: sum(len(ring) == 6 for ring in x.GetRingInfo().AtomRings()) >= 2,
        '>=_2_saturated_or_aromatic_carbon-only_ring_size_6': lambda x: sum(
            len(ring) == 6 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_nitrogen-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_heteroatom-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_carbon-only_ring_size_6': lambda x: sum(
            len(ring) == 6 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_nitrogen-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_heteroatom-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
            for i in range(len(ring))
            for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])]
            if bond is not None
        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_3_any_ring_size_6': lambda x: any(len(ring) == 6 for ring in  x.GetRingInfo().AtomRings()) >= 3,
        '>=_3_saturated_or_aromatic_carbon-only_ring_size_6': lambda x: any(
            len(ring) == 6 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_3_saturated_or_aromatic_nitrogen-containing_ring_size_6': lambda x: any(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_3_saturated_or_aromatic_heteroatom-containing_ring_size_6': lambda x: any(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_3_unsaturated_non-aromatic_carbon-only_ring_size_6': lambda x: any(
            len(ring) == 6 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_3_unsaturated_non-aromatic_nitrogen-containing_ring_size_6': lambda x: any(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_3_unsaturated_non-aromatic_heteroatom-containing_ring_size_6': lambda x: any(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        )>= 3,
        '>=_4_any_ring_size_6': lambda x: sum(len(ring) == 6 for ring in x.GetRingInfo().AtomRings()) >= 4,
        '>=_4_saturated_or_aromatic_carbon-only_ring_size_6': lambda x: sum(
            len(ring) == 6 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_4_saturated_or_aromatic_nitrogen-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_4_saturated_or_aromatic_heteroatom-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_4_unsaturated_non-aromatic_carbon-only_ring_size_6': lambda x: sum(
            len(ring) == 6 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_4_unsaturated_non-aromatic_nitrogen-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_4_unsaturated_non-aromatic_heteroatom-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 4,
        '>=_5_any_ring_size_6': lambda x: sum(len(ring) == 6 for ring in x.GetRingInfo().AtomRings()) >= 5,
        '>=_5_saturated_or_aromatic_carbon-only_ring_size_6': lambda x: sum(
            len(ring) == 6 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_5_saturated_or_aromatic_nitrogen-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_5_saturated_or_aromatic_heteroatom-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_5_unsaturated_non-aromatic_carbon-only_ring_size_6': lambda x: sum(
            len(ring) == 6 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_5_unsaturated_non-aromatic_nitrogen-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_5_unsaturated_non-aromatic_heteroatom-containing_ring_size_6': lambda x: sum(
            len(ring) == 6 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 5,
        '>=_1_any_ring_size_7': lambda x: any(len(ring) == 7 for ring in x.GetRingInfo().AtomRings()),
        '>=_1_saturated_or_aromatic_carbon-only_ring_size_7': lambda x: any(
            len(ring) == 7 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_nitrogen-containing_ring_size_7': lambda x: any(
            len(ring) == 7 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_heteroatom-containing_ring_size_7': lambda x: any(
            len(ring) == 7 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_carbon-only_ring_size_7': lambda x: any(
            len(ring) == 7 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_nitrogen-containing_ring_size_7': lambda x: any(
            len(ring) == 7 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_heteroatom-containing_ring_size_7': lambda x: any(
            len(ring) == 7 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_2_any_ring_size_7': lambda x: sum(len(ring) == 7 for ring in x.GetRingInfo().AtomRings()) >= 2,
        '>=_2_saturated_or_aromatic_carbon-only_ring_size_7': lambda x: sum(
            len(ring) == 7 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_nitrogen-containing_ring_size_7': lambda x: sum(
            len(ring) == 7 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_heteroatom-containing_ring_size_7': lambda x: sum(
            len(ring) == 7 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_carbon-only_ring_size_7': lambda x: sum(
            len(ring) == 7 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_nitrogen-containing_ring_size_7': lambda x: sum(
            len(ring) == 7 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_heteroatom-containing_ring_size_7': lambda x: sum(
            len(ring) == 7 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_1_any_ring_size_8': lambda x: any(len(ring) == 8 for ring in x.GetRingInfo().AtomRings()),
        '>=_1_saturated_or_aromatic_carbon-only_ring_size_8': lambda x: any(
            len(ring) == 8 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_nitrogen-containing_ring_size_8': lambda x: any(
            len(ring) == 8 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_heteroatom-containing_ring_size_8': lambda x: any(
            len(ring) == 8 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_carbon-only_ring_size_8': lambda x: any(
            len(ring) == 8 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_nitrogen-containing_ring_size_8': lambda x: any(
            len(ring) == 8 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_heteroatom-containing_ring_size_8': lambda x: any(
            len(ring) == 8 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_2_any_ring_size_8': lambda x: sum(len(ring) == 8 for ring in x.GetRingInfo().AtomRings()) >= 2,
        '>=_2_saturated_or_aromatic_carbon-only_ring_size_8': lambda x: sum(
            len(ring) == 8 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_nitrogen-containing_ring_size_8': lambda x: sum(
            len(ring) == 8 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_saturated_or_aromatic_heteroatom-containing_ring_size_8': lambda x: sum(
            len(ring) == 8 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_carbon-only_ring_size_8': lambda x: sum(
            len(ring) == 8 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_nitrogen-containing_ring_size_8': lambda x: sum(
            len(ring) == 8 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_2_unsaturated_non-aromatic_heteroatom-containing_ring_size_8': lambda x: sum(
            len(ring) == 8 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ) >= 2,
        '>=_1_any_ring_size_9': lambda x: any(len(ring) == 9 for ring in x.GetRingInfo().AtomRings()),
        '>=_1_saturated_or_aromatic_carbon-only_ring_size_9': lambda x: any(
            len(ring) == 9 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_nitrogen-containing_ring_size_9': lambda x: any(
            len(ring) == 9 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_heteroatom-containing_ring_size_9': lambda x: any(
            len(ring) == 9 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_carbon-only_ring_size_9': lambda x: any(
            len(ring) == 9 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_nitrogen-containing_ring_size_9': lambda x: any(
            len(ring) == 9 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_heteroatom-containing_ring_size_9': lambda x: any(
            len(ring) == 9 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_any_ring_size_10': lambda x: any(len(ring) == 10 for ring in x.GetRingInfo().AtomRings()),
        '>=_1_saturated_or_aromatic_carbon-only_ring_size_10': lambda x: any(
            len(ring) == 10 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_nitrogen-containing_ring_size_10': lambda x: any(
            len(ring) == 10 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_saturated_or_aromatic_heteroatom-containing_ring_size_10': lambda x: any(
            len(ring) == 10 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) > 0 or
                    all((bond.GetBondTypeAsDouble() == 1.0 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_carbon-only_ring_size_10': lambda x: any(
            len(ring) == 10 and all(x.GetAtomWithIdx(atom).GetSymbol() == 'C' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_nitrogen-containing_ring_size_10': lambda x: any(
            len(ring) == 10 and any(x.GetAtomWithIdx(atom).GetSymbol() == 'N' for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_unsaturated_non-aromatic_heteroatom-containing_ring_size_10': lambda x: any(
            len(ring) == 10 and any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring) and (
                    Chem.rdMolDescriptors.CalcNumAromaticRings(x) == 0 and
                    any((bond.GetBondTypeAsDouble() == 1.5 or bond.GetBondType() == Chem.BondType.AROMATIC)
                        for i in range(len(ring))
                        for bond in [x.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])]
                        if bond is not None
                        ))
            for ring in x.GetRingInfo().AtomRings()
        ),
        '>=_1_aromatic_ring': lambda x: sum(
            1 for ring in x.GetRingInfo().AtomRings()
            if rdMolDescriptors.CalcNumAromaticRings(Chem.PathToSubmol(x, list(ring))) > 0
        ) >= 1,
        '>=_1_hetero-aromatic_ring': lambda x: sum(
            1 for ring in x.GetRingInfo().AtomRings()
            if rdMolDescriptors.CalcNumAromaticRings(Chem.PathToSubmol(x, list(ring))) > 0 and
            any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring)
        ) >= 1,
        '>=_2_aromatic_rings': lambda x: sum(
            1 for ring in x.GetRingInfo().AtomRings()
            if rdMolDescriptors.CalcNumAromaticRings(Chem.PathToSubmol(x, list(ring))) > 0
        ) >= 2,
        '>=_2_hetero-aromatic_rings': lambda x: sum(
            1 for ring in x.GetRingInfo().AtomRings()
            if rdMolDescriptors.CalcNumAromaticRings(Chem.PathToSubmol(x, list(ring))) > 0 and
            any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring)
        ) >= 2,
        '>=_3_aromatic_rings': lambda x: sum(
            1 for ring in x.GetRingInfo().AtomRings()
            if rdMolDescriptors.CalcNumAromaticRings(Chem.PathToSubmol(x, list(ring))) > 0
        ) >= 3,
        '>=_3_hetero-aromatic_rings': lambda x: sum(
            1 for ring in x.GetRingInfo().AtomRings()
            if rdMolDescriptors.CalcNumAromaticRings(Chem.PathToSubmol(x, list(ring))) > 0
            and
            any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring)
        ) >= 3,
        '>=_4_aromatic_rings': lambda x: sum(
            1 for ring in x.GetRingInfo().AtomRings()
            if rdMolDescriptors.CalcNumAromaticRings(Chem.PathToSubmol(x, list(ring))) > 0
        ) >= 4,
        '>=_4_hetero-aromatic_rings': lambda x: sum(
            1 for ring in x.GetRingInfo().AtomRings()
            if rdMolDescriptors.CalcNumAromaticRings(Chem.PathToSubmol(x, list(ring))) > 0
            and
            any(x.GetAtomWithIdx(atom).GetSymbol() not in ['C', 'H'] for atom in ring)
        ) >= 4,
        'Li-H': lambda mol: any((atom1.GetSymbol() == 'Li' and atom2.GetSymbol() == 'H') or
                                (atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'Li')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'Li-Li': lambda mol: any(atom1.GetSymbol() == 'Li' and atom2.GetSymbol() == 'Li' for bond in mol.GetBonds()
                                 for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                       mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'Li-B': lambda mol: any((atom1.GetSymbol() == 'Li' and atom2.GetSymbol() == 'B') or
                                (atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'Li')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'Li-C': lambda mol: any((atom1.GetSymbol() == 'Li' and atom2.GetSymbol() == 'C') or
                                (atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'Li')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'Li-O': lambda mol: any((atom1.GetSymbol() == 'Li' and atom2.GetSymbol() == 'O') or
                                (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'Li')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'Li-F': lambda mol: any((atom1.GetSymbol() == 'Li' and atom2.GetSymbol() == 'F') or
                                (atom1.GetSymbol() == 'F' and atom2.GetSymbol() == 'Li')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'Li-P': lambda mol: any((atom1.GetSymbol() == 'Li' and atom2.GetSymbol() == 'P') or
                                (atom1.GetSymbol() == 'P' and atom2.GetSymbol() == 'Li')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'Li-S': lambda mol: any((atom1.GetSymbol() == 'Li' and atom2.GetSymbol() == 'S') or
                                (atom1.GetSymbol() == 'S' and atom2.GetSymbol() == 'Li')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'Li-Cl': lambda mol: any((atom1.GetSymbol() == 'Li' and atom2.GetSymbol() == 'Cl') or
                                 (atom1.GetSymbol() == 'Cl' and atom2.GetSymbol() == 'Li')
                                 for bond in mol.GetBonds()
                                 for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                       mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-H': lambda mol: any((atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'H') or
                               (atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'B')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-B': lambda mol: any(atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'B' for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-C': lambda mol: any((atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'C') or
                               (atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'B')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-N': lambda mol: any((atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'N') or
                               (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'B')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-O': lambda mol: any((atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'O') or
                               (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'B')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-F': lambda mol: any((atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'F') or
                               (atom1.GetSymbol() == 'F' and atom2.GetSymbol() == 'B')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-Si': lambda mol: any((atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'Si') or
                                (atom1.GetSymbol() == 'Si' and atom2.GetSymbol() == 'B')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-P': lambda mol: any((atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'P') or
                               (atom1.GetSymbol() == 'P' and atom2.GetSymbol() == 'B')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-S': lambda mol: any((atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'S') or
                               (atom1.GetSymbol() == 'S' and atom2.GetSymbol() == 'B')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-Cl': lambda mol: any((atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'Cl') or
                                (atom1.GetSymbol() == 'Cl' and atom2.GetSymbol() == 'B')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'B-Br': lambda mol: any((atom1.GetSymbol() == 'B' and atom2.GetSymbol() == 'Br') or
                                (atom1.GetSymbol() == 'Br' and atom2.GetSymbol() == 'B')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-H': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'H') or
                               (atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'C')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-C': lambda mol: any(atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C' for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-N': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N') or
                               (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'C')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-O': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'O') or
                               (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'C')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-F': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'F') or
                               (atom1.GetSymbol() == 'F' and atom2.GetSymbol() == 'C')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-Na': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'Na') or
                                (atom1.GetSymbol() == 'Na' and atom2.GetSymbol() == 'C')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-Mg': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'Mg') or
                                (atom1.GetSymbol() == 'Mg' and atom2.GetSymbol() == 'C')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-Al': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'Al') or
                                (atom1.GetSymbol() == 'Al' and atom2.GetSymbol() == 'C')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-Si': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'Si') or
                                (atom1.GetSymbol() == 'Si' and atom2.GetSymbol() == 'C')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-P': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'P') or
                               (atom1.GetSymbol() == 'P' and atom2.GetSymbol() == 'C')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-S': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'S') or
                               (atom1.GetSymbol() == 'S' and atom2.GetSymbol() == 'C')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-Cl': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'Cl') or
                                (atom1.GetSymbol() == 'Cl' and atom2.GetSymbol() == 'C')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-As': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'As') or
                                (atom1.GetSymbol() == 'As' and atom2.GetSymbol() == 'C')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-Se': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'Se') or
                                (atom1.GetSymbol() == 'Se' and atom2.GetSymbol() == 'C')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-Br': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'Br') or
                                (atom1.GetSymbol() == 'Br' and atom2.GetSymbol() == 'C')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C-I': lambda mol: any((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'I') or
                               (atom1.GetSymbol() == 'I' and atom2.GetSymbol() == 'C')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'N-H': lambda mol: any((atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'H') or
                               (atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'N')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'N-N': lambda mol: any(atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'N' for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'N-O': lambda mol: any((atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'O') or
                               (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'N')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'N-F': lambda mol: any((atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'F') or
                               (atom1.GetSymbol() == 'F' and atom2.GetSymbol() == 'N')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'N-Si': lambda mol: any((atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'Si') or
                                (atom1.GetSymbol() == 'Si' and atom2.GetSymbol() == 'N')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'N-P': lambda mol: any((atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'P') or
                               (atom1.GetSymbol() == 'P' and atom2.GetSymbol() == 'N')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'N-S': lambda mol: any((atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'S') or
                               (atom1.GetSymbol() == 'S' and atom2.GetSymbol() == 'N')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'N-Cl': lambda mol: any((atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'Cl') or
                                (atom1.GetSymbol() == 'Cl' and atom2.GetSymbol() == 'N')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'N-Br': lambda mol: any((atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'Br') or
                                (atom1.GetSymbol() == 'Br' and atom2.GetSymbol() == 'N')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'O-H': lambda mol: any((atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'H') or
                               (atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'O')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'O-O': lambda mol: any((atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'O')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'O-Mg': lambda mol: any((atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'Mg') or
                                (atom1.GetSymbol() == 'Mg' and atom2.GetSymbol() == 'O')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'O-Na': lambda mol: any((atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'Na') or
                                (atom1.GetSymbol() == 'Na' and atom2.GetSymbol() == 'O')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'O-Al': lambda mol: any((atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'Al') or
                                (atom1.GetSymbol() == 'Al' and atom2.GetSymbol() == 'O')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'O-Si': lambda mol: any((atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'Si') or
                                (atom1.GetSymbol() == 'Si' and atom2.GetSymbol() == 'O')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'O-P': lambda mol: any((atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'P') or
                               (atom1.GetSymbol() == 'P' and atom2.GetSymbol() == 'O')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'O-K': lambda mol: any((atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'K') or
                               (atom1.GetSymbol() == 'K' and atom2.GetSymbol() == 'O')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'F-P': lambda mol: any((atom1.GetSymbol() == 'F' and atom2.GetSymbol() == 'P') or
                               (atom1.GetSymbol() == 'P' and atom2.GetSymbol() == 'F')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'F-S': lambda mol: any((atom1.GetSymbol() == 'F' and atom2.GetSymbol() == 'S') or
                               (atom1.GetSymbol() == 'S' and atom2.GetSymbol() == 'F')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'Al-H': lambda mol: any((atom1.GetSymbol() == 'Al' and atom2.GetSymbol() == 'H') or
                                (atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'Al')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'Al-Cl': lambda mol: any((atom1.GetSymbol() == 'Al' and atom2.GetSymbol() == 'Cl') or
                                 (atom1.GetSymbol() == 'Cl' and atom2.GetSymbol() == 'Al')
                                 for bond in mol.GetBonds()
                                 for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                       mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'Si-H': lambda mol: any((atom1.GetSymbol() == 'Si' and atom2.GetSymbol() == 'H') or
                                (atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'Si')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'Si-Si': lambda mol: any((atom1.GetSymbol() == 'Si' and atom2.GetSymbol() == 'Si')
                                 for bond in mol.GetBonds()
                                 for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                       mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'Si-Cl': lambda mol: any((atom1.GetSymbol() == 'Si' and atom2.GetSymbol() == 'Cl') or
                                 (atom1.GetSymbol() == 'Cl' and atom2.GetSymbol() == 'Si')
                                 for bond in mol.GetBonds()
                                 for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                       mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'P-H': lambda mol: any((atom1.GetSymbol() == 'P' and atom2.GetSymbol() == 'H') or
                               (atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'P')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'P-P': lambda mol: any((atom1.GetSymbol() == 'P' and atom2.GetSymbol() == 'P')
                               for bond in mol.GetBonds()
                               for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                     mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'As-H': lambda mol: any((atom1.GetSymbol() == 'As' and atom2.GetSymbol() == 'H') or
                                (atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'As')
                                for bond in mol.GetBonds()
                                for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                      mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),

        'As-As': lambda mol: any((atom1.GetSymbol() == 'As' and atom2.GetSymbol() == 'As')
                                 for bond in mol.GetBonds()
                                 for atom1, atom2 in [(mol.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                                                       mol.GetAtomWithIdx(bond.GetEndAtomIdx()))]),
        'C(~Br)(~C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Br') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~Br)(~C)(~C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Br') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~Br)(~H)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Br') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~Br)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Br') == 1 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~Br)(:N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Br') == 1 and
            any(neighbor.GetSymbol() == 'N' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~C)(~C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 3
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~C)(~C)(~C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 4
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~C)(~C)(~H)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 3 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~C)(~C)(~N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 3 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~C)(~C)(~O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 3 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~C)(~H)(~N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~C)(~H)(~O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~C)(~N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~C)(~O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~Cl)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Cl') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~Cl)(~H)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Cl') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~H)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~H)(~N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~H)(~O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~H)(~O)(~O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~H)(~P)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'P') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~H)(~S)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'S') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~I)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'I') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~S)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'S') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(~Si)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Si') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~C)(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~C)(:C)(:N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors()) and
            any(neighbor.GetSymbol() == 'N' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~C)(:N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            any(neighbor.GetSymbol() == 'N' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~C)(:N)(:N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~Cl)(~Cl)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Cl') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~Cl)(~H)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Cl') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~Cl)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Cl') == 1 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~F)(~F)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'F') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~F)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'F') == 1 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~H)(~N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~H)(~O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~H)(~O)(~O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~H)(~S)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'S') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~H)(~Si)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'Si') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~H)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~H)(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~H)(:C)(:N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors()) and
            any(neighbor.GetSymbol() == 'N' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~H)(:N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            any(neighbor.GetSymbol() == 'N' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~H)(~H)(~H)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 3
            for atom in mol.GetAtoms()
        ),
        'C(~N)(~N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~N)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~N)(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~N)(:C)(:N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors()) and
            any(neighbor.GetSymbol() == 'N' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~N)(:N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1
            for atom in mol.GetAtoms()
        ),
        'C(~O)(~O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~O)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(~O)(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'C(~S)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'S') == 1 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'C(:C)(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 3
            for atom in mol.GetAtoms()
        ),
        'C(:C)(:C)(:N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2 and
            any(neighbor.GetSymbol() == 'N' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(:C)(:N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 1 and
            any(neighbor.GetSymbol() == 'N' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'C(:C)(:N)(:N)': lambda mol: any(
                    atom.GetSymbol() == 'C' and
                    sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2 and
                    sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1
                    for atom in mol.GetAtoms()
                ),
        'C(:N)(:N)': lambda mol: any(
                    atom.GetSymbol() == 'C' and
                    sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 2
                    for atom in mol.GetAtoms()
                ),
        'N(~C)(~C)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'N(~C)(~C)(~C)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 3
            for atom in mol.GetAtoms()
        ),
        'N(~C)(~C)(~H)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1
            for atom in mol.GetAtoms()
        ),
        'N(~C)(~H)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1
            for atom in mol.GetAtoms()
        ),
        'N(~C)(~H)(~N)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1
            for atom in mol.GetAtoms()
        ),
        'N(~C)(~O)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1
            for atom in mol.GetAtoms()
        ),
        'N(~C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 1 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'N(~C)(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'N(~H)(~N)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 0 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N') == 1
            for atom in mol.GetAtoms()
        ),
        'N(~H)(:C)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 0 and
            any(neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'N(~H)(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 0 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'N(~O)(~O)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 2
            for atom in mol.GetAtoms()
        ),
        'N(~O)(:O)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1 and
            any(neighbor.GetSymbol() == 'O' for neighbor in atom.GetNeighbors())
            for atom in mol.GetAtoms()
        ),
        'N(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'N(:C)(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 3
            for atom in mol.GetAtoms()
        ),
        'O(~C)(~C)': lambda mol: any(
            atom.GetSymbol() == 'O' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'O(~C)(~H)': lambda mol: any(
            atom.GetSymbol() == 'O' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 0
            for atom in mol.GetAtoms()
        ),
        'O(~C)(~P)': lambda mol: any(
            atom.GetSymbol() == 'O' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'P') == 0
            for atom in mol.GetAtoms()
        ),
        'O(~H)(~S)': lambda mol: any(
            atom.GetSymbol() == 'O' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 0 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'S') == 1
            for atom in mol.GetAtoms()
        ),
        'O(:C)(:C)': lambda mol: any(
            atom.GetSymbol() == 'O' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'P(~C)(~C)': lambda mol: any(
            atom.GetSymbol() == 'P' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'P(~O)(~O)': lambda mol: any(
            atom.GetSymbol() == 'P' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 2
            for atom in mol.GetAtoms()
        ),
        'S(~C)(~C)': lambda mol: any(
            atom.GetSymbol() == 'S' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'S(~C)(~H)': lambda mol: any(
            atom.GetSymbol() == 'S' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H') == 0
            for atom in mol.GetAtoms()
        ),
        'S(~C)(~O)': lambda mol: any(
            atom.GetSymbol() == 'S' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 1 and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O') == 1
            for atom in mol.GetAtoms()
        ),
        'Si(~C)(~C)': lambda mol: any(
            atom.GetSymbol() == 'Si' and
            sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'C') == 2
            for atom in mol.GetAtoms()
        ),
        'C=C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'C' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 2) > 0
            for atom in mol.GetAtoms()
        ),
        'C#C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'C' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 3) > 0
            for atom in mol.GetAtoms()
        ),
        'C=N': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'N' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 2) > 0
            for atom in mol.GetAtoms()
        ),
        'C#N': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'N' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 3) > 0
            for atom in mol.GetAtoms()
        ),
        'C=O': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 2) > 0
            for atom in mol.GetAtoms()
        ),
        'C=S': lambda mol: any(
            atom.GetSymbol() == 'C' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'S' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 2) > 0
            for atom in mol.GetAtoms()
        ),
        'N=N': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'N' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 2) > 0
            for atom in mol.GetAtoms()
        ),
        'N=O': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 2) > 0
            for atom in mol.GetAtoms()
        ),
        'N=P': lambda mol: any(
            atom.GetSymbol() == 'N' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'P' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 1) > 0
            for atom in mol.GetAtoms()
        ),
        'P=O': lambda mol: any(
            atom.GetSymbol() == 'P' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 2) > 0
            for atom in mol.GetAtoms()
        ),
        'P=P': lambda mol: any(
            atom.GetSymbol() == 'P' and
            sum(1 for neighbor in atom.GetNeighbors() if
                neighbor.GetSymbol() == 'P' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                        neighbor.GetIdx()).GetBondTypeAsDouble() == 2) > 0
            for atom in mol.GetAtoms()
        ),
        'C(1': lambda mol: 0,
        'C(2': lambda mol: 0,
        'C(3': lambda mol: 0,
        'C(-C)(-C)(=C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),
        'C(-C)(-C)(=N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),
        'C(-C)(-C)(=O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),
        'C(-C)(-Cl)(=O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'Cl' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            ) and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),
        'C(-C)(-H)(=C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'H' or neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-C)(-H)(=N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'H' or neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-C)(-H)(=O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'H' or neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-C)(-N)(=C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C(-C)(-N)(=N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'N' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-C)(-N)(=O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'N' or neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-C)(-O)(=O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'O' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-C)(=C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C(-C)(=N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C(-C)(=O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C(-Cl)(=O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'Cl' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C(-H)(-N)(=C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'H' or neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-H)(=C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'H' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-H)(=N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'H' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-H)(=O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'H' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-N)(=C)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-N)(=N)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-N)(=O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'O' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'C(-O)(=O)': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'O' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'N(-C)(=C)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N(-C)(=O)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'O' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'N(-O)(=O)': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'O' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'P(-O)(=O)': lambda mol: any(
            atom.GetSymbol() == 'P' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'O' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'S(-C)(=O)': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S(-O)(=O)': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            ) and
            all(
                neighbor.GetSymbol() == 'O' for neighbor in atom.GetNeighbors()
                if mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() != 2
            )
            for atom in mol.GetAtoms()
        ),

        'S(=O)(=O)': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            all(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C=N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C=O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'O' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N:C-S-[': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C-C=C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O=S-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    neighbor2.GetSymbol() == 'C' and
                    len(neighbor2.GetNeighbors()) == 2
                    for neighbor2 in neighbor.GetNeighbors()
                ) and
                any(
                    neighbor2.GetSymbol() == 'C' and
                    mol.GetBondBetweenAtoms(neighbor.GetIdx(), neighbor2.GetIdx()).GetBondTypeAsDouble() == 1
                    for neighbor2 in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N':lambda mol: 0 ,
        'C=N-N-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=S-C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S-S-C:C': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'S' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C:C-C=C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S:C:C:C': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C:N:C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S-C:N:C': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S:C:C:N': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S-C=N-C': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-O-C=C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'O' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-N-C:C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S-C=N-[': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S-C-S-C': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'S' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C:S:C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-S-C:C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C:N-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-S-C:C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C:N:C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N:C:C:N': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C:N:N': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C=N-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C=N-[': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C-S-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'S' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-C=C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-N:C-[': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C:O:C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C:C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C:N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-N-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N:N-C-[': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C:N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'N' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C=C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                neighbor.GetSymbol() == 'C' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N-C:C:N': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'N' and
                    len(n.GetNeighbors()) == 2
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-S-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'C' and
                    len(n.GetNeighbors()) == 2
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cl-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'C' and
                    len(n.GetNeighbors()) == 2
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C=C-[': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'C' and
                    len(n.GetNeighbors()) == 2 and
                    any(
                        nn.GetSymbol() == '[' and
                        len(nn.GetNeighbors()) == 0
                        for nn in n.GetNeighbors()
                    )
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cl-C:C-[': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'C' and
                    len(n.GetNeighbors()) == 2 and
                    any(
                        nn.GetSymbol() == '[' and
                        len(nn.GetNeighbors()) == 0
                        for nn in n.GetNeighbors()
                    )
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N:C:N-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'N' and
                    len(n.GetNeighbors()) == 2 and
                    any(
                        nn.GetSymbol() == 'C' and
                        len(nn.GetNeighbors()) == 2
                        for nn in n.GetNeighbors()
                    )
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cl-C:C-O': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'O' and
                    len(n.GetNeighbors()) == 1
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C:N:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    n.GetSymbol() == 'N' and
                    len(n.GetNeighbors()) == 2 and
                    any(
                        nn.GetSymbol() == 'C' and
                        len(nn.GetNeighbors()) == 2
                        for nn in n.GetNeighbors()
                    )
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-S-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    n.GetSymbol() == 'S' and
                    len(n.GetNeighbors()) == 2 and
                    any(
                        nn.GetSymbol() == 'C' and
                        len(nn.GetNeighbors()) == 2
                        for nn in n.GetNeighbors()
                    )
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S=C-N-C': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'N' and
                    len(n.GetNeighbors()) == 2 and
                    any(
                        nn.GetSymbol() == 'C' and
                        len(nn.GetNeighbors()) == 2
                        for nn in n.GetNeighbors()
                    )
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Br-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'Br' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'C' and
                    len(n.GetNeighbors()) == 2
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        '[0':lambda mol: 0,
        'S=C-N-[': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'N' and
                    len(n.GetNeighbors()) == 2
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-[As]-O-[': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'As' and
                len(neighbor.GetNeighbors()) == 1 and
                any(
                    n.GetSymbol() == 'O' and
                    len(n.GetNeighbors()) == 1
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S:C:C-[': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'C' and
                    len(n.GetNeighbors()) == 2
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-N-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'C' and
                    len(n.GetNeighbors()) == 2 and
                    any(
                        nn.GetSymbol() == 'C'
                        for nn in n.GetNeighbors()
                    )
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-N-C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    n.GetSymbol() == 'C' and
                    len(n.GetNeighbors()) == 2 and
                    any(
                        nn.GetSymbol() == 'C'
                        for nn in n.GetNeighbors()
                    )
                    for n in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        '[':lambda mol: 0,
        'N-N-C-N': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            ) and
            any(
                next_neighbor.GetSymbol() == 'N' and
                len(next_neighbor.GetNeighbors()) == 2
                for next_neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-N-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N=C-N-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C=C-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C:N-C-[': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C-N-N-[': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N:C:C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C'
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C-C=C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        '[As]-C:C-[': lambda mol: any(
            atom.GetSymbol() == '[As]' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cl-C:C-Cl': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'Cl'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C:C:N-[': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        '[11':lambda mol: 0 ,
        'Cl-C-C-Cl': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                all(next_neighbor.GetSymbol() == 'Cl' for next_neighbor in neighbor.GetNeighbors())
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N:C-C:C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(next_neighbor.GetSymbol() == 'C' for next_neighbor in neighbor.GetNeighbors())
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(next_neighbor.GetSymbol() == 'C' for next_neighbor in neighbor.GetNeighbors())
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S-C:C-[': lambda mol: any(
    atom.GetSymbol() == 'S' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 2 and
        any(
            next_neighbor.GetSymbol() == 'S' and
            len(next_neighbor.GetNeighbors()) == 2 and
            any(
                next_next_neighbor.GetSymbol() == 'C' and
                len(next_next_neighbor.GetNeighbors()) == 2
                for next_next_neighbor in next_neighbor.GetNeighbors()
            )
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
        for atom in mol.GetAtoms()
    ),

    'S-C:C-N': lambda mol: any(
        atom.GetSymbol() == 'S' and
        len(atom.GetNeighbors()) == 2 and
        any(
            neighbor.GetSymbol() == 'C' and
            len(neighbor.GetNeighbors()) == 2 and
            any(
                next_neighbor.GetSymbol() == 'C' and
                len(next_neighbor.GetNeighbors()) == 2 and
                any(
                    next_next_neighbor.GetSymbol() == 'N'
                    for next_next_neighbor in next_neighbor.GetNeighbors()
                )
                for next_neighbor in neighbor.GetNeighbors()
            )
            for neighbor in atom.GetNeighbors()
        )
        for atom in mol.GetAtoms()
    ),

        'S-C:C-O': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'O'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'N'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'O'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N=C-C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N=C-C-[': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == '['
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-N-C-[': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == '['
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C-[': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == '['
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'N'
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C-[': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    any(
                        next_next_neighbor.GetSymbol() == '['
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O-C:C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    any(
                        next_next_neighbor.GetSymbol() == 'N'
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'O'
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C:C-[': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == '['
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C:C-N': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'N'
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C:C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N-C-C:C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cl-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cl-C-C-O': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 1 and
                    any(
                        last_neighbor.GetSymbol() == 'O'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C:C-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O=C-C=C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Br-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'Br' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N=C-C=C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C=C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N:C-O-[': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 1
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O=N-C:C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O-C-N-[': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == '['
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N-C-N-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cl-C-C=O': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 1 and
                    any(
                        last_neighbor.GetSymbol() == 'O'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Br-C-C=O': lambda mol: any(
            atom.GetSymbol() == 'Br' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 1 and
                    any(
                        last_neighbor.GetSymbol() == 'O'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O-C-O-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 1 and
                    any(
                        last_neighbor.GetSymbol() == 'C'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C=C-C=C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C:C-O-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 1 and
                    any(
                        last_neighbor.GetSymbol() == 'C'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O-C-C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'N'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O-C-C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'O'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N0': lambda mol: any(
            atom.GetSymbol() == 'N'
            for atom in mol.GetAtoms()
        ),
        'N-C-C-N': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'N'
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C:C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        '[111':lambda mol: 0,
        'O=C-N-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N:C:N:C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C=C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C:C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'O' and
                        len(final_neighbor.GetNeighbors()) == 1
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N=C-C:C-[': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 1
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C:C-N-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 4
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C:C-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 4
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        final_neighbor.GetSymbol() == 'N' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        final_neighbor.GetSymbol() == 'O' and
                        len(final_neighbor.GetNeighbors()) == 1
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 4 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 4
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cl-C:C-O-C': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 1 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 3
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C:C-C=C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C:C-N-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 4
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-S-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 4 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 4
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C:C-O-[': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'O' and
                        len(final_neighbor.GetNeighbors()) == 1
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C=O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        final_neighbor.GetSymbol() == 'O' and
                        len(final_neighbor.GetNeighbors()) == 1
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C:C-O-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 1 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 3
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C:C-O-[': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 1 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 3
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cl-C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 4 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 4
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 4 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 4
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C-C-C-N': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 4 and
                    any(
                        final_neighbor.GetSymbol() == 'N' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-O-C-C=C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'O' and
                len(neighbor.GetNeighbors()) == 1 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 4 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C:C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 4 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 4
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N=C-N-C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 4
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O=C-C-C:C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cl-C:C:C-C': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        '[1':lambda mol:0,
        'N-C:C:C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C:C:C-N': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            ) and any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-N-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'N' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C:C:C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            ) and any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-O-C-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'O' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            ) and any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-O-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'O' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            ) and any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C-C-C:C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-C-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            ) and any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cl-C-C-N-C': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'N' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-O-C-O-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'O' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            ) and any(
                neighbor.GetSymbol() == 'O' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C-C-N-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'N' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C-O-C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-N-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-O-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C-C-O-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'O' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C:C:N:N:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            ) and any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-C-O-[': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'O' and
                        len(last_neighbor.GetNeighbors()) == 1
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C:C-C-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C=C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C:C-O-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C:C:C:N': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'N' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            ) and any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-O-C:C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            ) and any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O=C-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C'
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C:C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'N'
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O=C-C:C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'O' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-O-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'O' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=[As]-C:C:C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'As' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-N-C-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 4 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'S-C:C:C-N': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'N' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C-O-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'O' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C-O-[': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'O'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-O-C:C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-N-C-N-[': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 4 and
                    any(
                        next_next_neighbor.GetSymbol() == 'N' and
                        len(next_next_neighbor.GetNeighbors()) == 2
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-N-C-N-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C'
                    for next_neighbor in neighbor.GetNeighbors()
                    )
                    for neighbor in atom.GetNeighbors()
                )
            for atom in mol.GetAtoms()
        ),

        'O-C-C-C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2 and
                        any(
                            final_neighbor.GetSymbol() == 'N' and
                            len(final_neighbor.GetNeighbors()) == 2
                            for final_neighbor in next_next_neighbor.GetNeighbors()
                        )
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C-C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2 and
                        any(
                            final_neighbor.GetSymbol() == 'O' and
                            len(final_neighbor.GetNeighbors()) == 2
                            for final_neighbor in next_next_neighbor.GetNeighbors()
                        )
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C=C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C-C=C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2 and
                        any(
                            final_neighbor.GetSymbol() == 'C' and
                            len(final_neighbor.GetNeighbors()) == 2
                            for final_neighbor in next_next_neighbor.GetNeighbors()
                        )
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C-C=O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C' and
                        len(next_next_neighbor.GetNeighbors()) == 2 and
                        any(
                            final_neighbor.GetSymbol() == 'O' and
                            len(final_neighbor.GetNeighbors()) == 2
                            for final_neighbor in next_next_neighbor.GetNeighbors()
                        )
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        '[2':lambda mol:0,
        'C-C=N-N-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'N' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-N-C-[': lambda mol: any(
        atom.GetSymbol() == 'O' and
        len(atom.GetNeighbors()) == 2 and
        any(
            neighbor.GetSymbol() == 'C' and
            len(neighbor.GetNeighbors()) == 2 and
            any(
                next_neighbor.GetSymbol() == 'N' and
                len(next_neighbor.GetNeighbors()) == 2 and
                any(
                    next_next_neighbor.GetSymbol() == 'C' and
                    any(
                        last_neighbor.GetSymbol() == '['
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_next_neighbor in next_neighbor.GetNeighbors()
                )
                for next_neighbor in neighbor.GetNeighbors()
            )
            for neighbor in atom.GetNeighbors()
        )
        for atom in mol.GetAtoms()
    ),

        'O=C-N-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2 and
                        any(
                            another_neighbor.GetSymbol() == 'C' and
                            len(another_neighbor.GetNeighbors()) == 2
                            for another_neighbor in final_neighbor.GetNeighbors()
                        )
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-N-C-[': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'N'
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O=C-N-C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2 and
                        any(
                            another_neighbor.GetSymbol() == 'N' and
                            len(another_neighbor.GetNeighbors()) == 2
                            for another_neighbor in final_neighbor.GetNeighbors()
                        )
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=N-C:C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2 and
                        any(
                            another_neighbor.GetSymbol() == 'N' and
                            len(another_neighbor.GetNeighbors()) == 2
                            for another_neighbor in final_neighbor.GetNeighbors()
                        )
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=N-C:C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2 and
                        any(
                            another_neighbor.GetSymbol() == 'O' and
                            len(another_neighbor.GetNeighbors()) == 2
                            for another_neighbor in final_neighbor.GetNeighbors()
                        )
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-N-C=O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2 and
                        any(
                            another_neighbor.GetSymbol() == 'O' and
                            len(another_neighbor.GetNeighbors()) == 2
                            for another_neighbor in final_neighbor.GetNeighbors()
                        )
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C:C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C:C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        final_neighbor.GetSymbol() == 'N' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C:C:C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        final_neighbor.GetSymbol() == 'O' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'N-C-N-C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2 and
                        any(
                            another_neighbor.GetSymbol() == 'C' and
                            len(another_neighbor.GetNeighbors()) == 2
                            for another_neighbor in final_neighbor.GetNeighbors()
                        )
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C-C:C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 3
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-N-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-N-C:C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-S-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'S' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()),
        'O-C-C-N-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'N' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C-C=C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 1
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O-C-O-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'O' and
                    len(next_neighbor.GetNeighbors()) == 1 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O-C-C-O-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'O' and
                        len(last_neighbor.GetNeighbors()) == 1 and
                        any(
                            final_neighbor.GetSymbol() == 'C' and
                            len(final_neighbor.GetNeighbors()) == 2
                            for final_neighbor in last_neighbor.GetNeighbors()
                        )
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O-C-C-O-[': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'O' and
                        len(last_neighbor.GetNeighbors()) == 1 and
                        any(
                            final_neighbor.GetSymbol() == '[' and
                            len(final_neighbor.GetNeighbors()) == 0
                            for final_neighbor in last_neighbor.GetNeighbors()
                        )
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C-C=C-C=C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N-C:C-C-C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 1
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C=C-C-O-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        last_neighbor.GetSymbol() == 'O' and
                        len(last_neighbor.GetNeighbors()) == 1 and
                        any(
                            final_neighbor.GetSymbol() == 'C' and
                            len(final_neighbor.GetNeighbors()) == 2
                            for final_neighbor in last_neighbor.GetNeighbors()
                        )
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C=C-C-O-[': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        last_neighbor.GetSymbol() == 'O' and
                        len(last_neighbor.GetNeighbors()) == 1 and
                        any(
                            final_neighbor.GetSymbol() == '[' and
                            len(final_neighbor.GetNeighbors()) == 0
                            for final_neighbor in last_neighbor.GetNeighbors()
                        )
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'C-C:C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 1
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cl-C:C-C=O': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2 and
                        any(
                            final_neighbor.GetSymbol() == 'O' and
                            len(final_neighbor.GetNeighbors()) == 1
                            for final_neighbor in last_neighbor.GetNeighbors()
                        )
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Br-C:C:C-C': lambda mol: any(
            atom.GetSymbol() == 'Br' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 3 and
                        any(
                            final_neighbor.GetSymbol() == 'C' and
                            len(final_neighbor.GetNeighbors()) == 1
                            for final_neighbor in last_neighbor.GetNeighbors()
                        )
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O=C-C=C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O=C-C=C-[': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2 and
                        any(
                            final_neighbor.GetSymbol() == '[' and
                            len(final_neighbor.GetNeighbors()) == 0
                            for final_neighbor in last_neighbor.GetNeighbors()
                        )
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'O=C-C=C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 2 and
                        any(
                            final_neighbor.GetSymbol() == 'N' and
                            len(final_neighbor.GetNeighbors()) == 2
                            for final_neighbor in last_neighbor.GetNeighbors()
                        )
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N-C-N-C:C': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'N' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 3 and
                        any(
                            final_neighbor.GetSymbol() == 'C' and
                            len(final_neighbor.GetNeighbors()) == 1
                            for final_neighbor in last_neighbor.GetNeighbors()
                        )
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Br-C-C-C:C': lambda mol: any(
            atom.GetSymbol() == 'Br' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        last_neighbor.GetSymbol() == 'C' and
                        len(last_neighbor.GetNeighbors()) == 1
                        for last_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'N4':lambda mol:0,
        'C-C=C-C:C': lambda mol: any(
    atom.GetSymbol() == 'C' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 3 and
        any(
            next_neighbor.GetSymbol() == 'C' and
            len(next_neighbor.GetNeighbors()) == 1
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'C-C-C=C-C': lambda mol: any(
    atom.GetSymbol() == 'C' and
    len(atom.GetNeighbors()) == 3 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 3 and
        any(
            next_neighbor.GetSymbol() == 'C' and
            len(next_neighbor.GetNeighbors()) == 2
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'C-C-C-C-C-C': lambda mol: any(
    atom.GetSymbol() == 'C' and
    len(atom.GetNeighbors()) == 6 and
    all(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 2
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'O-C-C-C-C-C': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 1 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 5 and
        all(
            next_neighbor.GetSymbol() == 'C' and
            len(next_neighbor.GetNeighbors()) == 2
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'O-C-C-C-C-O': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 4 and
        any(
            next_neighbor.GetSymbol() == 'C' and
            len(next_neighbor.GetNeighbors()) == 2
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'O-C-C-C-C-N': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 4 and
        any(
            next_neighbor.GetSymbol() == 'N' and
            len(next_neighbor.GetNeighbors()) == 2
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'N-C-C-C-C-C': lambda mol: any(
    atom.GetSymbol() == 'N' and
    len(atom.GetNeighbors()) == 1 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 5 and
        all(
            next_neighbor.GetSymbol() == 'C' and
            len(next_neighbor.GetNeighbors()) == 2
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'O=C-C-C-C-C': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 5 and
        all(
            next_neighbor.GetSymbol() == 'C' and
            len(next_neighbor.GetNeighbors()) == 2
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'O=C-C-C-C-N': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 4 and
        any(
            next_neighbor.GetSymbol() == 'N' and
            len(next_neighbor.GetNeighbors()) == 2
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'O=C-C-C-C-O': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 4 and
        any(
            next_neighbor.GetSymbol() == 'O' and
            len(next_neighbor.GetNeighbors()) == 1
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'O=C-C-C-C=O': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 4 and
        any(
            next_neighbor.GetSymbol() == 'O' and
            len(next_neighbor.GetNeighbors()) == 2
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'C-C-C-C-C-C-C': lambda mol: any(
    atom.GetSymbol() == 'C' and
    len(atom.GetNeighbors()) == 7 and
    all(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 2
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'O-C-C-C-C-C-C': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 1 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 6 and
        all(
            next_neighbor.GetSymbol() == 'C' and
            len(next_neighbor.GetNeighbors()) == 2
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),
        'O-C-C-C-C-C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C'
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C-C-C-C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'N'
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C'
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-C-C-O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'O'
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-C-C=O': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'O'
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-C-C-N': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'N'
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-C-C-C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C'
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-C-C-C-C(C)-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3 and
                    any(
                        final_neighbor.GetSymbol() == 'C'
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C-C-C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'C'
                        for final_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

'O-C-C-C-C-C(C)-C': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 6 and
        any(
            next_neighbor.GetSymbol() == 'C' and
            len(next_neighbor.GetNeighbors()) == 3
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'O-C-C-C-C-C-O-C': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 5 and
        any(
            next_neighbor.GetSymbol() == 'O' and
            len(next_neighbor.GetNeighbors()) == 1
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),

'O-C-C-C-C-C(O)-C': lambda mol: any(
    atom.GetSymbol() == 'O' and
    len(atom.GetNeighbors()) == 2 and
    any(
        neighbor.GetSymbol() == 'C' and
        len(neighbor.GetNeighbors()) == 6 and
        any(
            next_neighbor.GetSymbol() == 'O' and
            len(next_neighbor.GetNeighbors()) == 1
            for next_neighbor in neighbor.GetNeighbors()
        )
        for neighbor in atom.GetNeighbors()
    )
    for atom in mol.GetAtoms()
),
        'O-C-C-C-C-C-N-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'N' and
                        len(final_neighbor.GetNeighbors()) == 2
                        for final_neighbor in neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O-C-C-C-C-C(N)-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        final_neighbor.GetSymbol() == 'N' and
                        len(final_neighbor.GetNeighbors()) == 3
                        for final_neighbor in neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-C-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-C-C(O)-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3
                    for next_neighbor in neighbor.GetNeighbors()
                ) and
                any(
                    final_neighbor.GetSymbol() == 'O' and
                    len(final_neighbor.GetNeighbors()) == 1
                    for final_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-C-C(=O)-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3
                    for next_neighbor in neighbor.GetNeighbors()
                ) and
                any(
                    final_neighbor.GetSymbol() == 'O' and
                    len(final_neighbor.GetNeighbors()) == 1
                    for final_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'O=C-C-C-C-C(N)-C': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3
                    for next_neighbor in neighbor.GetNeighbors()
                ) and
                any(
                    final_neighbor.GetSymbol() == 'N' and
                    len(final_neighbor.GetNeighbors()) == 2
                    for final_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C(C)-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C(C)-C-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C-C(C)-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C(C)(C)-C-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'C-C(C)-C(C)-C': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 3
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1ccc(C)cc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1ccc(O)cc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'O'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cc1ccc(S)cc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'S'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cc1ccc(N)cc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'N'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cc1ccc(Cl)cc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'Cl'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Cc1ccc(Br)cc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'Br'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Oc1ccc(O)cc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'O'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Oc1ccc(S)cc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'S'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Oc1ccc(N)cc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'N'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Oc1ccc(Cl)cc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'Cl'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Oc1ccc(Br)cc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'Br'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Sc1ccc(S)cc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'S'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Sc1ccc(N)cc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'N'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Sc1ccc(Cl)cc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'Cl'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Sc1ccc(Br)cc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'Br'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Nc1ccc(N)cc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'N'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Nc1ccc(Cl)cc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'Cl'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Nc1ccc(Br)cc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'Br'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Clc1ccc(Cl)cc1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'Cl'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'Clc1ccc(Br)cc1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                any(
                    child_neighbor.GetSymbol() == 'Br'
                    for child_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Brc1ccc(Br)cc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1cc(C)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1cc(O)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1cc(S)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1cc(N)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1cc(Cl)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1cc(O)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1cc(S)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1cc(N)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1cc(Cl)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1cc(S)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1cc(N)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1cc(Cl)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居碳原子有3个邻居（可能是苯环上的一部分）
                all(  # 该碳原子的所有邻居都满足特定条件
                    next_neighbor.GetSymbol() == 'C' or  # 是碳原子
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                     len(next_neighbor.GetBonds()) == 1)  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Nc1cc(N)ccc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Nc1cc(Cl)ccc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Nc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Clc1cc(Cl)ccc1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Clc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Brc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'Br' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(C)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(O)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(S)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(N)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(Cl)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(Br)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1c(O)cccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1c(S)cccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1c(N)cccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1c(Cl)cccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1c(Br)cccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1c(S)cccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1c(N)cccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1c(Cl)cccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1c(Br)cccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Nc1c(N)cccc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Nc1c(Cl)cccc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Nc1c(Br)cccc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3 and
                all(
                    next_neighbor.GetSymbol() == 'C' or
                    (next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and
                     len(next_neighbor.GetBonds()) == 1)
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Clc1c(Cl)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Clc1c(Br)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Brc1c(Br)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(C)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居有3个邻居（苯环的一部分）
                all(
                    next_neighbor.GetSymbol() == 'C' or  # 该邻居是碳原子
                    next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                    len(next_neighbor.GetBonds()) == 1  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(O)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'O' and  # 其中一个邻居是氧原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氧原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'S' and  # 其中一个邻居是硫原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（硫原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'N' and  # 其中一个邻居是氮原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（氮原子通常有3个键）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'Cl' and  # 其中一个邻居是氯原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'Br' and  # 其中一个邻居是溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CCC(O)CC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居（氧原子通常有2个键，但在环上可能只连接1个）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（苯环的一部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'SC1CCC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'S' and  # 查找硫原子
            len(atom.GetNeighbors()) == 1 and  # 该硫原子有1个邻居（硫原子通常有2个键，但在环上可能只连接1个）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（苯环的一部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'NC1CCC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'N' and  # 查找氮原子
            len(atom.GetNeighbors()) == 3 and  # 该氮原子有3个邻居（氮原子通常有3个键）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（苯环的一部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'ClC1CCC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and  # 查找氯原子
            len(atom.GetNeighbors()) == 1 and  # 该氯原子有1个邻居（氯原子不连接其他原子）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（苯环的一部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'ClC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and  # 查找氯原子
            len(atom.GetNeighbors()) == 1 and  # 该氯原子有1个邻居（氯原子不连接其他原子）
            any(
                neighbor.GetSymbol() == 'Br' and  # 其中一个邻居是溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'BrC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'Br' and  # 查找溴原子
            len(atom.GetNeighbors()) == 1 and  # 该溴原子有1个邻居（溴原子不连接其他原子）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（苯环的一部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CC(C)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（环的部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CC(O)CCC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居（氧原子在环上通常只连接1个其他原子）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（环的部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CC(S)CCC1': lambda mol: any(
            atom.GetSymbol() == 'S' and  # 查找硫原子
            len(atom.GetNeighbors()) == 1 and  # 该硫原子有1个邻居（硫原子在环上通常只连接1个其他原子）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（环的部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CC(N)CCC1': lambda mol: any(
            atom.GetSymbol() == 'N' and  # 查找氮原子
            len(atom.GetNeighbors()) == 3 and  # 该氮原子有3个邻居（氮原子通常有3个键）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（环的部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),'CC1CCC(C)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3 and  # 该邻居有3个邻居（苯环的一部分）
                all(
                    next_neighbor.GetSymbol() == 'C' or  # 该邻居是碳原子
                    next_neighbor.GetSymbol() in ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br'] and  # 或者是允许的其他原子
                    len(next_neighbor.GetBonds()) == 1  # 且只连接了一个键（可能是氢原子或其他取代基）
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'CC1CCC(O)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'O' and  # 其中一个邻居是氧原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氧原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'CC1CCC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'S' and  # 其中一个邻居是硫原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（硫原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'CC1CCC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'N' and  # 其中一个邻居是氮原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（氮原子通常有3个键）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'CC1CCC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'Cl' and  # 其中一个邻居是氯原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'CC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'Br' and  # 其中一个邻居是溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'OC1CCC(O)CC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居（氧原子通常有2个键，但在环上可能只连接1个）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（苯环的一部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'SC1CCC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'S' and  # 查找硫原子
            len(atom.GetNeighbors()) == 1 and  # 该硫原子有1个邻居（硫原子通常有2个键，但在环上可能只连接1个）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（苯环的一部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'NC1CCC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'N' and  # 查找氮原子
            len(atom.GetNeighbors()) == 3 and  # 该氮原子有3个邻居（氮原子通常有3个键）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（苯环的一部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'ClC1CCC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and  # 查找氯原子
            len(atom.GetNeighbors()) == 1 and  # 该氯原子有1个邻居（氯原子不连接其他原子）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（苯环的一部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'ClC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and  # 查找氯原子
            len(atom.GetNeighbors()) == 1 and  # 该氯原子有1个邻居（氯原子不连接其他原子）
            any(
                neighbor.GetSymbol() == 'Br' and  # 其中一个邻居是溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'BrC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'Br' and  # 查找溴原子
            len(atom.GetNeighbors()) == 1 and  # 该溴原子有1个邻居（溴原子不连接其他原子）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（苯环的一部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'CC1CC(C)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（环的部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'CC1CC(O)CCC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居（氧原子在环上通常只连接1个其他原子）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（环的部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'CC1CC(S)CCC1': lambda mol: any(
            atom.GetSymbol() == 'S' and  # 查找硫原子
            len(atom.GetNeighbors()) == 1 and  # 该硫原子有1个邻居（硫原子在环上通常只连接1个其他原子）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（环的部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
'CC1CC(N)CCC1': lambda mol: any(
            atom.GetSymbol() == 'N' and  # 查找氮原子
            len(atom.GetNeighbors()) == 3 and  # 该氮原子有3个邻居（氮原子通常有3个键）
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（环的部分）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1cc(Cl)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'Cl' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'Br' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1cc(O)ccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'O' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1cc(S)ccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1cc(N)ccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1cc(Cl)ccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'Cl' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'Br' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1cc(S)ccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1cc(N)ccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1cc(Cl)ccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'Cl' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'Br' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Nc1cc(N)ccc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Nc1cc(Cl)ccc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'Cl' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Nc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'Br' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Clc1cc(Cl)ccc1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'Cl' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Clc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'Br' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Brc1cc(Br)ccc1': lambda mol: any(
            atom.GetSymbol() == 'Br' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'Br' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(C)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(O)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'O' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(S)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(S)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'S' and  # 其中一个邻居是硫原子
                len(neighbor.GetNeighbors()) == 2  # 该邻居有2个邻居（硫原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(N)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'N' and  # 其中一个邻居是氮原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（氮原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(Cl)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'Cl' and  # 其中一个邻居是氯原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(Br)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'Br' and  # 其中一个邻居是溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1c(O)cccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 4  # 该邻居有4个邻居（碳原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1c(S)cccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'S' and  # 其中一个邻居是硫原子
                len(neighbor.GetNeighbors()) == 2  # 该邻居有2个邻居（硫原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1c(N)cccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'N' and  # 其中一个邻居是氮原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（氮原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1c(Cl)cccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'Cl' and  # 其中一个邻居是氯原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Oc1c(Br)cccc1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'Br' and  # 其中一个邻居是溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1c(S)cccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and  # 查找硫原子
            len(atom.GetNeighbors()) == 2 and  # 该硫原子有2个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 4  # 该邻居有4个邻居（碳原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1c(N)cccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and  # 查找硫原子
            len(atom.GetNeighbors()) == 2 and  # 该硫原子有2个邻居
            any(
                neighbor.GetSymbol() == 'N' and  # 其中一个邻居是氮原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（氮原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1c(Cl)cccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and  # 查找硫原子
            len(atom.GetNeighbors()) == 2 and  # 该硫原子有2个邻居
            any(
                neighbor.GetSymbol() == 'Cl' and  # 其中一个邻居是氯原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Sc1c(Br)cccc1': lambda mol: any(
            atom.GetSymbol() == 'S' and  # 查找硫原子
            len(atom.GetNeighbors()) == 2 and  # 该硫原子有2个邻居
            any(
                neighbor.GetSymbol() == 'Br' and  # 其中一个邻居是溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(S)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'S' and  # 其中一个邻居是硫原子
                len(neighbor.GetNeighbors()) == 2  # 该邻居有2个邻居（硫原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(N)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'N' and  # 其中一个邻居是氮原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（氮原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(Cl)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'Cl' and  # 其中一个邻居是氯原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'Cc1c(Br)cccc1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'Br' and  # 其中一个邻居是溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CCC(O)CC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 4  # 该邻居有4个邻居（碳原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CCC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'S' and  # 其中一个邻居是硫原子
                len(neighbor.GetNeighbors()) == 2  # 该邻居有2个邻居（硫原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CCC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'N' and  # 其中一个邻居是氮原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（氮原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CCC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'Cl' and  # 其中一个邻居是氯原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'Br' and  # 其中一个邻居是溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(C)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 4  # 该邻居有4个邻居（碳原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(O)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'O' and  # 其中一个邻居是氧原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氧原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'S' and  # 其中一个邻居是硫原子
                len(neighbor.GetNeighbors()) == 2  # 该邻居有2个邻居（硫原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'N' and  # 其中一个邻居是氮原子
                len(neighbor.GetNeighbors()) == 3  # 该邻居有3个邻居（氮原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'Cl' and  # 其中一个邻居是氯原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() == 'Br' and  # 其中一个邻居是溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CCC(O)CC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'C' and  # 其中一个邻居是碳原子
                len(neighbor.GetNeighbors()) == 4  # 该邻居有4个邻居（碳原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CCC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'O' and  # 查找氧原子
            len(atom.GetNeighbors()) == 1 and  # 该氧原子有1个邻居
            any(
                neighbor.GetSymbol() == 'S' and  # 其中一个邻居是硫原子
                len(neighbor.GetNeighbors()) == 2  # 该邻居有2个邻居（硫原子连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CCC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'N'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1CCC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1CCC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1CCC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S', 'N'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1CCC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S', 'Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S', 'Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'NC1CCC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['N'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'NC1CCC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['N', 'Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'NC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['N', 'Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'ClC1CCC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'ClC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'BrC1CCC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1CC(C)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1CC(O)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1CC(S)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1CC(N)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['N'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1CC(Cl)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1CC(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1CC(O)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O'] and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1CC(S)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'S'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1CC(N)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'N'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1CC(Cl)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1CC(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1CC(S)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'SC1CC(N)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S', 'N'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1CC(Cl)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S', 'Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1CC(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S', 'Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'NC1CC(N)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['N'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'NC1CC(Cl)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['N', 'Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'NC1CC(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['N', 'Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'ClC1CC(Cl)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'ClC1CC(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'BrC1CC(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1C(C)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['C'] and
                len(neighbor.GetNeighbors()) == 4
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1C(O)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O'] and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1C(S)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S'] and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1C(N)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['N'] and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1C(Cl)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1C(Br)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1C(O)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O'] and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1C(S)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'S'] and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1C(N)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'N'] and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1C(Cl)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1C(Br)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['O', 'Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1C(S)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S'] and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1C(N)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S', 'N'] and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1C(Cl)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S', 'Cl'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1C(Br)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['S', 'Br'] and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'NC1C(N)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() in ['N'] and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'NC1C(Cl)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'NC1C(Br)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'ClC1C(Cl)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'ClC1C(Br)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'BrC1C(Br)CCCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CC(C)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CC(O)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1CC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CC(O)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'OC1CC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'SC1CC(S)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'SC1CC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'SC1CC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'SC1CC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'NC1CC(N)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'NC1CC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'NC1CC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'ClC1CC(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'ClC1CC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'BrC1CC(Br)CC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1C(C)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1C(O)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and  # 查找碳原子
            len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
            any(
                neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
                len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),
        'CC1C(S)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1C(N)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1C(Cl)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'Cl' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'CC1C(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'C' and
            len(atom.GetNeighbors()) == 4 and
            any(
                neighbor.GetSymbol() == 'Br' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1C(O)CCC1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1C(S)CCC1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1C(N)CCC1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1C(Cl)CCC1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'Cl' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'OC1C(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'O' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'Br' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1C(S)CCC1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'S' and
                len(neighbor.GetNeighbors()) == 2
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1C(N)CCC1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1C(Cl)CCC1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'Cl' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'SC1C(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'S' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'Br' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'NC1C(N)CCC1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'N' and
                len(neighbor.GetNeighbors()) == 3
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'NC1C(Cl)CC1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 2 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 2 and
                any(
                    next_neighbor.GetSymbol() == 'C' and
                    len(next_neighbor.GetNeighbors()) == 2 and
                    any(
                        next_next_neighbor.GetSymbol() == 'C'
                        for next_next_neighbor in next_neighbor.GetNeighbors()
                    )
                    for next_neighbor in neighbor.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'NC1C(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'N' and
            len(atom.GetNeighbors()) == 3 and
            any(
                neighbor.GetSymbol() == 'Br' and
                len(neighbor.GetNeighbors()) == 1
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'ClC1C(Cl)CCC1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    neighbor.GetSymbol() == 'Cl' and
                    len(neighbor.GetNeighbors()) == 1
                    for neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

        'ClC1C(Br)CCC1': lambda mol: any(
            atom.GetSymbol() == 'Cl' and
            len(atom.GetNeighbors()) == 1 and
            any(
                neighbor.GetSymbol() == 'C' and
                len(neighbor.GetNeighbors()) == 4 and
                any(
                    neighbor.GetSymbol() == 'Br' and
                    len(neighbor.GetNeighbors()) == 1
                    for neighbor in atom.GetNeighbors()
                )
                for neighbor in atom.GetNeighbors()
            )
            for atom in mol.GetAtoms()
        ),

    'BrC1C(Br)CCC1': lambda mol: any(
        atom.GetSymbol() == 'C' and  # 查找碳原子
        len(atom.GetNeighbors()) == 4 and  # 该碳原子有4个邻居
        any(
            neighbor.GetSymbol() in ['Cl', 'Br'] and  # 其中一个邻居是氯或溴原子
            len(neighbor.GetNeighbors()) == 1  # 该邻居有1个邻居（氯或溴原子不连接其他原子）
            for neighbor in atom.GetNeighbors()
        )
        for atom in mol.GetAtoms()
    )
}  # <<< GLOBAL_CONDITIONS 字典定义结束


def check_conditions(mol):
    """
    检查 SMILES 是否符合指定的化学结构描述符条件。
    如果 mol 为 None (SMILES 无效)，则返回所有条件为 0 的字典。
    """
    if mol is None:
        # 如果 mol 对象为 None，表示 SMILES 无效，无法计算描述符。
        # 为所有条件返回 0，使用全局字典的键来确保完整性。
        return {key: 0 for key in GLOBAL_CONDITIONS.keys()}

    # 添加氢原子 (仅当 mol 有效时才执行)
    mol = Chem.AddHs(mol)

    result = {}
    # 遍历全局定义的条件，计算每个描述符
    for key, condition_func in GLOBAL_CONDITIONS.items():  # <<< 这里使用 GLOBAL_CONDITIONS
        try:
            # 执行条件函数并存储结果
            result[key] = 1 if condition_func(mol) else 0
        except Exception as e:
            # 捕获在计算特定条件时可能发生的任何RDKit或其他错误
            # 对于无法计算的情况，将其设置为0，并打印警告信息
            print(f"Warning: Could not calculate condition '{key}' for molecule. Setting to 0. Error: {e}")
            result[key] = 0
    return result


def process_smiles(input_file, output_file):
    """
    读取CSV文件，为每条SMILES字符串检查条件，并将结果添加到新的CSV文件中。
    """
    df = pd.read_csv(input_file)
    smiles_column = df.columns[2]  # 假设 SMILES 在第三列

    results = []
    # 使用tqdm为循环添加进度条，方便跟踪处理进度
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing SMILES"):
        smiles = row[smiles_column]

        # 尝试将SMILES转换为RDKit分子对象
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # 如果RDKit无法解析SMILES，打印警告并跳过计算
            print(
                f"Warning: RDKit could not parse SMILES: '{smiles}' at row {index}. Skipping descriptor calculation for this row.")
            # 调用 check_conditions 并传入 None，它会返回一个全0的字典
            condition_results = check_conditions(None)
        else:
            # 如果SMILES有效，则正常计算描述符
            condition_results = check_conditions(mol)

        results.append(condition_results)

    # 将结果列表转换为DataFrame
    results_df = pd.DataFrame(results)

    # 将原始数据DataFrame与新计算的描述符DataFrame合并
    df_final = pd.concat([df, results_df], axis=1)

    # 将最终的DataFrame写入新的CSV文件
    df_final.to_csv(output_file, index=False)
    print(f"Processing complete. Results saved to '{output_file}'")


# 使用示例
input_csv = 'predict_all_np.csv'
output_csv = 'drug_with_conditions_predict_all_np.csv'

# 确保输入文件存在
if not os.path.exists(input_csv):
    print(f"Error: Input file '{input_csv}' not found. Please check the file path.")
else:
    process_smiles(input_csv, output_csv)