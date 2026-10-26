def boltz_safe(smiles: str):
    """utils.molecules.is_boltz_safe_smiles, same logic: (ok, reason)."""
    from rdkit.Chem import AllChem
    try:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return False, "RDKit failed to parse SMILES"
        mol = AllChem.AddHs(mol)
        canonical_order = AllChem.CanonicalRankAtoms(mol)
        for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
            atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
            if len(atom_name) > 4:
                return False, f"Atom name would exceed 4 chars: {atom_name}"
        return True, None
    except Exception as error:
        return False, f"Boltz safety check failed: {error}"
