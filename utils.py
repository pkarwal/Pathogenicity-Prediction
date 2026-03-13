import re

def parse_cds_mutation(cds):

    pattern = r"c\.(\d+)([A-Z])>([A-Z])"
    match = re.match(pattern, cds)

    if not match:
        raise ValueError("Invalid CDS mutation format")

    pos = int(match.group(1))
    from_base = match.group(2)
    to_base = match.group(3)

    return pos, from_base, to_base


def parse_aa_mutation(aa):

    pattern = r"p\.([A-Z])(\d+)([A-Z])"
    match = re.match(pattern, aa)

    if not match:
        raise ValueError("Invalid AA mutation format")

    from_aa = match.group(1)
    pos = int(match.group(2))
    to_aa = match.group(3)

    return pos, from_aa, to_aa