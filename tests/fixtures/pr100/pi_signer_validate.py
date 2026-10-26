# Pure validator function extracted from the installed Pi signer, module SHA
# 4065f06134a32e214c61627ee6c004d5f598b73a519258773ce1e8d388bd85fe.
# No signing or authentication implementation is included.
SUBMISSION_DELIMITER = '|'
SUBMISSION_SEPARATOR = ','
NULL_TOKEN = '~'
CANONICAL_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

def validate_plaintext(plaintext: str) -> str:
    """Validate a fully framed molecule/nanobody submission without changing it.

    The combined publisher has already assembled validator wire text. It must
    be encrypted byte-for-byte: running it through ``submission_message`` would
    turn ``mols|sequence`` into ``~|MOLS|SEQUENCE`` and invalidate the payload.
    """
    if not isinstance(plaintext, str) or not plaintext:
        raise ValueError("refusing to commit empty plaintext")
    if not plaintext.isascii():
        raise ValueError("a submission must be ascii")
    if any(character.isspace() for character in plaintext):
        raise ValueError("a submission may not contain whitespace")
    if plaintext.count(SUBMISSION_DELIMITER) != 1:
        raise ValueError("a submission must contain exactly one section delimiter")

    molecule_part, sequence = plaintext.split(SUBMISSION_DELIMITER, 1)
    if not sequence or set(sequence) - CANONICAL_AMINO_ACIDS:
        raise ValueError("sequence must be nonempty canonical amino acids")

    if molecule_part == NULL_TOKEN:
        return plaintext

    molecule_ids = molecule_part.split(SUBMISSION_SEPARATOR)
    if len(molecule_ids) != 20:
        raise ValueError("molecule section must contain exactly 20 identifiers")
    if len(set(molecule_ids)) != 20:
        raise ValueError("molecule identifiers must be unique")

    # Kept local to validation so --help and argparse remain usable on a host
    # with only this deployment script installed.
    from compound_miner.types import parse_reaction_name

    reactions: set[int] = set()
    for molecule_id in molecule_ids:
        try:
            reaction, _ = parse_reaction_name(molecule_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid molecule identifier: {molecule_id!r}"
            ) from error
        if reaction not in range(1, 45):
            raise ValueError(f"unsupported molecule reaction: {reaction}")
        reactions.add(reaction)
    if len(reactions) != 1:
        raise ValueError("all molecule identifiers must use the same reaction")
    return plaintext
