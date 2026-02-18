import csv
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os

def highlight_atoms(mol, atom_indices, color=(1, 0.5, 0.5)):
    """Highlight specific atoms in a molecule drawing."""
    highlight_atoms = atom_indices
    highlight_colors = {idx: color for idx in atom_indices}
    return highlight_atoms, highlight_colors

def verify_conversion(smiles, psmiles, row_idx, output_dir='verification_images'):
    """
    Verify pSMILES conversion by drawing both structures and comparing.
    
    Args:
        smiles: Original SMILES with [Au] and [Cu]
        psmiles: Converted pSMILES with [*]
        row_idx: Row index for naming
        output_dir: Directory to save images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse original SMILES
    mol_orig = Chem.MolFromSmiles(smiles)
    if mol_orig is None:
        print(f"Row {row_idx}: Failed to parse original SMILES")
        return False
    
    # Parse pSMILES
    mol_psmiles = Chem.MolFromSmiles(psmiles)
    if mol_psmiles is None:
        print(f"Row {row_idx}: Failed to parse pSMILES")
        return False
    
    # Find Au and Cu atoms in original
    au_cu_indices = []
    au_cu_neighbors = []
    for atom in mol_orig.GetAtoms():
        if atom.GetSymbol() in ['Au', 'Cu']:
            au_cu_indices.append(atom.GetIdx())
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1:
                au_cu_neighbors.append(neighbors[0].GetIdx())
    
    # Find [*] atoms in pSMILES
    star_indices = []
    star_neighbors = []
    for atom in mol_psmiles.GetAtoms():
        if atom.GetAtomicNum() == 0:  # [*] has atomic number 0
            star_indices.append(atom.GetIdx())
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 1:
                star_neighbors.append(neighbors[0].GetIdx())
    
    # Verify counts match
    if len(au_cu_indices) != len(star_indices):
        print(f"Row {row_idx}: Mismatch - {len(au_cu_indices)} Au/Cu atoms but {len(star_indices)} [*] atoms")
        return False
    
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol_orig)
    AllChem.Compute2DCoords(mol_psmiles)
    
    # Draw original with Au/Cu highlighted
    highlight_orig, colors_orig = highlight_atoms(mol_orig, au_cu_indices, (1, 0.7, 0.7))
    img_orig = Draw.MolToImage(mol_orig, size=(400, 300), 
                                highlightAtoms=highlight_orig,
                                highlightAtomColors=colors_orig)
    
    # Draw pSMILES with [*] highlighted
    highlight_psmiles, colors_psmiles = highlight_atoms(mol_psmiles, star_indices, (0.7, 1, 0.7))
    img_psmiles = Draw.MolToImage(mol_psmiles, size=(400, 300),
                                   highlightAtoms=highlight_psmiles,
                                   highlightAtomColors=colors_psmiles)
    
    # Save images
    img_orig.save(f'{output_dir}/row_{row_idx}_original.png')
    img_psmiles.save(f'{output_dir}/row_{row_idx}_psmiles.png')
    
    # Verify structure
    # Count non-dummy atoms
    orig_non_dummy = sum(1 for atom in mol_orig.GetAtoms() if atom.GetSymbol() not in ['Au', 'Cu'])
    psmiles_non_star = sum(1 for atom in mol_psmiles.GetAtoms() if atom.GetAtomicNum() != 0)
    
    if orig_non_dummy != psmiles_non_star:
        print(f"Row {row_idx}: Atom count mismatch - Original has {orig_non_dummy} non-dummy atoms, pSMILES has {psmiles_non_star} non-[*] atoms")
        return False
    
    return True

def main():
    """Verify pSMILES conversions for sample rows."""
    input_file = 'data/htpmd_with_psmiles.csv'
    
    # Read CSV
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f'Loaded {len(rows)} rows from {input_file}')
    print('Verifying conversions for first 10 rows...\n')
    
    success_count = 0
    fail_count = 0
    
    # Verify first 10 rows
    for i in range(min(10, len(rows))):
        row = rows[i]
        smiles = row['smiles']
        psmiles = row['psmiles']
        
        print(f'Row {i}:')
        print(f'  Original: {smiles}')
        print(f'  pSMILES:  {psmiles}')
        
        if verify_conversion(smiles, psmiles, i):
            print(f'  ✓ Verification passed')
            success_count += 1
        else:
            print(f'  ✗ Verification failed')
            fail_count += 1
        print()
    
    print(f'\nVerification Summary:')
    print(f'  Success: {success_count}')
    print(f'  Failed:  {fail_count}')
    print(f'\nImages saved to: verification_images/')
    print(f'  - Original SMILES (Au/Cu highlighted in red)')
    print(f'  - pSMILES ([*] highlighted in green)')

if __name__ == '__main__':
    main()
