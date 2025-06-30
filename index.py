def can_skip(entry: list, other_doc: int) -> bool:
    """
    Check if a skip is possible in a posting list entry.

    A skip is possible if the entry has valid skip pointer information
    and the docID at the skip position is less than or equal to the
    other document's ID.

    Args:
        entry (list): A list in the form [docID, skip_index, docID_at_skip].
        other_doc (int): The target document ID to compare against.

    Returns:
        bool: True if skipping is possible, False otherwise.
    """
    return entry[1] is not None and entry[2] is not None and entry[2] <= other_doc


def intersect_skip(A: list, B: list) -> list:
    """
    Intersect two sorted posting lists that contain skip pointers.

    Each list must consist of entries that are lists of exactly three elements:
    [docID (int), skip_index (int), docID_at_skip (int)].

    Args:
        A (list): First posting list. Must be a list of [int, int, int] entries.
        B (list): Second posting list. Must be a list of [int, int, int] entries.

    Returns:
        list: A list of document IDs (int) present in both A and B.

    Raises:
        ValueError: If an entry in A or B is not a list of three integers.
    """
    i = 0
    j = 0
    matches = []
    while i < len(A) and j < len(B):
        if A[i][0] == B[j][0]:
            matches.append(A[i][0])
            i += 1
            j += 1
        elif A[i][0] < B[j][0]:
            if can_skip(A[i], B[j][0]):
                i = A[i][1]
                continue
            i += 1
        elif A[i][0] > B[j][0]:
            if can_skip(B[j], A[i][0]):
                j = B[j][1]
                continue
            j += 1
        else:
            raise Exception("Something went wrong...")
    return matches


def in_range(A: list[int], B: list[int], rng: int) -> bool:
    """
    Check if any positions in two sorted lists fall within a given range.

    Compares values from two lists and returns True if at least one
    pair of elements (one from each list) differs by at most `rng`.

    Args:
        A (list[int]): First list of positions.
        B (list[int]): Second list of positions.
        rng (int): Maximum allowed difference between position values.

    Returns:
        bool: True if any positions are within `rng`, False otherwise.
    """
    i = 0
    j = 0
    while i < len(A) and j < len(B):
        if abs(A[i] - B[j]) <= rng:
            return True
        elif A[i] < B[j]:
            i += 1
        elif A[i] > B[j]:
            j += 1
        else:
            raise Exception("Something went wrong...")
    return False


def intersect_range(A: list, B: list, rng: int) -> list:
    """
    Intersect two posting lists with document-internal positional constraints.

    Each posting list entry must be of the form [docID, positions],
    where positions is a sorted list of integers. A match occurs if
    both lists have the same docID and there is at least one pair
    of positions (one from each list) within the given `rng`.

    Args:
        A (list): First posting list as [docID, positions].
        B (list): Second posting list as [docID, positions].
        rng (int): Maximum allowed position difference for a match.

    Returns:
        list: List of document IDs that match within the specified range.
    """
    i = 0
    j = 0
    matches = []
    while i < len(A) and j < len(B):
        docA, posA = A[i]
        docB, posB = B[j]
        if docA == docB:
            if in_range(posA, posB, rng):
                matches.append(docA)
            i += 1
            j += 1
        elif docA < docB:
            i += 1
        else:
            j += 1
    return matches


#Posting lists with skip pointers. 
#Entries take the form [docID, index to skip to, docID at that index]
times_skip = [[2, 3, 16], [12, None, None], [15, None, None], [16, 6, 27], [17, None, None], [23, None, None], [27, None, None]]
square_skip = [[3, 2, 12], [8, None, None], [12, 4, 23], [19, None, None], [23, None, None]]


#Posting lists with document-internal positional information.
#Entries take the form [docID, [pos1, pos2, ...]]
times_range = [[2, [15, 128]], [12, [6, 45, 89, 942]], [15, [13]], [16, [1276, 1500]], [17, [13, 89, 90]], [23, [17, 64]], [27, [456, 629]]]
square_range = [[3, [65, 90]], [8, [67, 94]], [12, [3]], [19, [18, 81, 1881]], [23, [63]]]

words_dict_skip = {}
words_dict_skip['times'] = times_skip
words_dict_skip['square'] = square_skip
print(intersect_skip(words_dict_skip['times'], words_dict_skip['square']))

words_dict_range = {}
words_dict_range['times'] = times_range
words_dict_range['square'] = square_range
print(intersect_range(words_dict_range['times'], words_dict_range['square'], 3))