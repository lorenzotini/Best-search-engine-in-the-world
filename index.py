# index.py

def can_skip(entry: list, other_doc: int) -> bool:
    """Return True if other_doc appears in this entry’s skip-pointer list."""
    return any(item[0] == other_doc for item in entry)

def intersect_skip(A: list, B: list) -> list:
    """Intersect two skip-pointer lists (each item = [doc_id, skip_to, skip_doc])."""
    i, j, out = 0, 0, []
    while i < len(A) and j < len(B):
        if A[i][0] == B[j][0]:
            out.append(A[i])
            i += 1
            j += 1
        elif A[i][0] < B[j][0]:
            i += 1
        else:
            j += 1
    return out

def in_range(A: list[int], B: list[int], rng: int) -> bool:
    """Return True if any position in A is within rng of some position in B."""
    i, j = 0, 0
    while i < len(A) and j < len(B):
        if abs(A[i] - B[j]) <= rng:
            return True
        if A[i] < B[j]:
            i += 1
        else:
            j += 1
    return False

def intersect_range(A: list[int], B: list[int], rng: int) -> list[int]:
    """Return all positions in A for which there’s a position in B within rng."""
    i, j, out = 0, 0, []
    while i < len(A) and j < len(B):
        if abs(A[i] - B[j]) <= rng:
            out.append(A[i])
            i += 1
        elif A[i] < B[j]:
            i += 1
        else:
            j += 1
    return out
