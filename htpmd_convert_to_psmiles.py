import csv
from rdkit import Chem

def smiles_to_psmiles(smiles):
    """
    Convert SMILES with [Au] and [Cu] dummy atoms to pSMILES with [*] connection points.
    
    Args:
        smiles: SMILES string with [Au] and [Cu] as connection point markers
        
    Returns:
        pSMILES string with [*] replacing the dummy atoms
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Find Au and Cu atoms
    dummy_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ['Au', 'Cu']:
            dummy_atoms.append(atom.GetIdx())
    
    # Get the neighbor of each dummy atom (connection point)
    connection_points = []
    for dummy_idx in dummy_atoms:
        atom = mol.GetAtomWithIdx(dummy_idx)
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            print(f"Warning: Dummy atom {atom.GetSymbol()} at index {dummy_idx} has {len(neighbors)} neighbors (expected 1)")
            return None
        connection_points.append(neighbors[0].GetIdx())
    
    # Create editable molecule
    emol = Chem.EditableMol(mol)
    
    # Remove dummy atoms (remove in reverse order to maintain indices)
    for dummy_idx in sorted(dummy_atoms, reverse=True):
        emol.RemoveAtom(dummy_idx)
    
    # Get the modified molecule
    new_mol = emol.GetMol()
    
    # Adjust connection point indices after removal
    # When we remove atoms, indices shift down
    adjusted_connection_points = []
    for cp_idx in connection_points:
        # Count how many dummy atoms were removed before this connection point
        removed_before = sum(1 for d_idx in dummy_atoms if d_idx < cp_idx)
        adjusted_idx = cp_idx - removed_before
        adjusted_connection_points.append(adjusted_idx)
    
    # Add [*] atoms at connection points
    emol2 = Chem.EditableMol(new_mol)
    star_indices = []
    for cp_idx in adjusted_connection_points:
        star_idx = emol2.AddAtom(Chem.Atom(0))  # Atomic number 0 = [*]
        star_indices.append(star_idx)
        emol2.AddBond(cp_idx, star_idx, Chem.BondType.SINGLE)
    
    final_mol = emol2.GetMol()
    
    # Generate pSMILES
    psmiles = Chem.MolToSmiles(final_mol)
    
    return psmiles


def convert_csv():
    """
    Read htpmd.csv and add a psmiles column after the smiles column.
    """
    input_file = 'data/htpmd.csv'
    output_file = 'data/htpmd_with_psmiles.csv'
    
    # Read input CSV
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    print(f'Processing {len(rows)} rows...')
    
    # Find smiles column index
    smiles_idx = fieldnames.index('smiles')
    
    # Insert psmiles column after smiles
    new_fieldnames = fieldnames[:smiles_idx+1] + ['psmiles'] + fieldnames[smiles_idx+1:]
    
    # Convert each row
    success_count = 0
    fail_count = 0
    
    for i, row in enumerate(rows):
        smiles = row['smiles']
        psmiles = smiles_to_psmiles(smiles)
        
        if psmiles is None:
            print(f'Failed to convert row {i}: {smiles}')
            fail_count += 1
            row['psmiles'] = ''
        else:
            row['psmiles'] = psmiles
            success_count += 1
        
        if (i + 1) % 1000 == 0:
            print(f'Processed {i + 1}/{len(rows)} rows...')
    
    # Write output CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f'\nConversion complete!')
    print(f'Success: {success_count} rows')
    print(f'Failed: {fail_count} rows')
    print(f'Output written to: {output_file}')
    
    # Show a few examples
    print('\nExample conversions:')
    for i in range(min(5, len(rows))):
        print(f'\nRow {i}:')
        print(f'  SMILES:  {rows[i]["smiles"]}')
        print(f'  pSMILES: {rows[i]["psmiles"]}')


if __name__ == '__main__':
    convert_csv()
