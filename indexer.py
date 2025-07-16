from collections import defaultdict
import pickle
import math
import os


class Indexer:
    def __init__(self):
        self.path_to_TFs = 'data/tfs.pkl'
        self.path_to_IDFs = 'data/idfs.pkl'
        self.path_to_crawled_data = 'data/crawled_data.pkl'
        self.path_to_posting_lists = 'data/posting_list.pkl'
        
        self.crawled_data = self._load(self.path_to_crawled_data)
        self.skip_dict, self.pos_index_dict = self._load(self.path_to_posting_lists)


    def run(self):
        self._index_documents()
        self._build_TF()
        self._build_IDF()


    def get_candidates(self, query, use_proximity=False, proximity_range=3):
        """
        Get candidate document IDs for the given query.

        Args:
            query (list of str): List of query terms (already preprocessed)
            use_proximity (bool): If True, use _intersect_range for positional constraints.
            proximity_range (int): Max distance between terms if use_proximity is True.

        Returns:
            list of int: Candidate document IDs.
        """
        if not query:
            return []

        query = [term.lower() for term in query]  # Normalize

        if use_proximity:
            # Positional intersection (for phrase search / proximity search)
            if query[0] not in self.pos_index_dict:
                return []

            candidates = self.pos_index_dict[query[0]]
            
            for term in query[1:]:
                if term not in self.pos_index_dict:
                    return []
                candidates = self._intersect_range(candidates, self.pos_index_dict[term], proximity_range)
                # After first iteration, candidates becomes a list of docIDs only
                # For further _intersect_range calls, we need to reconstruct position lists
                if not candidates:
                    return []
                candidates = [
                    [doc_id, [pos for pos in self.pos_index_dict[term] if pos[0] == doc_id][0][1]]
                    for doc_id in candidates
                ]
            
            return [doc_id for doc_id, _ in candidates]

        else:
            # Standard AND search using skip pointers
            if query[0] not in self.skip_dict:
                return []

            candidates = self.skip_dict[query[0]]

            for term in query[1:]:
                if term not in self.skip_dict:
                    return []
                candidates = self._intersect_skip(candidates, self.skip_dict[term])
                if not candidates:
                    return []

                # After _intersect_skip, candidates is a list of docIDs
                # For the next iteration, convert back to [docID, skip_index, docID_at_skip]
                # But since we don't have skip pointers anymore, wrap it without skips
                candidates = [[doc_id, None, None] for doc_id in candidates]

            return [entry[0] for entry in candidates]


    def _index_documents(self):
        print('Indexing...')
        self._build_skip_pointers(self.crawled_data)
        self._build_positional_index(self.crawled_data)
        
        print("Saving posting lists...")
        with open(self.path_to_posting_lists, "wb") as f:
            pickle.dump((self.skip_dict, self.pos_index_dict), f)
        print("Done.")


    def _build_skip_pointers(self, crawled_data):
        token_to_ids = {}
        skip_dict = {}

        # Process every document
        for doc_id, doc_data in crawled_data.items():
            tokens = doc_data.get("tokens")
            
            # TODO fix this check
            if tokens is None:
                continue

            # For every token, create a list containing the IDs of the documents in which the token is present
            for token in tokens:
                if token not in token_to_ids:    # First time we see this term -> initialize it
                    token_to_ids[token] = [doc_id]
                elif doc_id in token_to_ids[token]:
                    continue
                else:
                    token_to_ids[token].append(doc_id)
        
        # Add skip pointers
        for token in token_to_ids:
            current_ids = token_to_ids[token]
            
            # Insert a pointer every pointer_freq entry
            pointer_freq = math.ceil(math.sqrt(len(token_to_ids[token])))

            skip_pointers = []
            for i, id in enumerate(current_ids):
                index_to_skip = None
                doc_at_that_index = None
                if i % pointer_freq == 0:
                    index_to_skip = i + pointer_freq
                    if index_to_skip >= len(current_ids):
                        index_to_skip = len(current_ids) - 1
                    doc_at_that_index = current_ids[index_to_skip]
                skip_pointers.append([id, index_to_skip, doc_at_that_index])
            
            skip_dict[token] = skip_pointers
        
        self.skip_dict = skip_dict

    
    def _build_positional_index(self, crawled_data):

        index = defaultdict(lambda: defaultdict(list))  # token -> {doc_id: [positions]}

        for doc_id, doc_data in crawled_data.items():
            tokens = doc_data.get("tokens")
            # TODO fix this check
            if tokens is None:
                continue
            for position, token in enumerate(tokens):
                index[token][doc_id].append(position)

        # Convert inner dicts to lists for final output format
        final_index = {}
        for token, doc_dict in index.items():
            final_index[token] = [[doc_id, positions] for doc_id, positions in doc_dict.items()]

        self.pos_index_dict = final_index


    def _build_TF(self):
        tf_list = []
        for doc_id, doc_data in self.crawled_data.items():            
            # build term-frequency dictionary
            bow = {}
            tokens = doc_data.get("tokens")
            # TODO fix this check
            if tokens is None:
                continue
            for word in tokens:
                bow[word] = bow.get(word, 0) + 1
            
            tf_list.append(bow)
        # Save data in file
        with open(self.path_to_TFs, "wb") as f:
            pickle.dump(tf_list, f)


    #Estimate inverse document frequencies based on a corpus of documents.
    def _build_IDF(self):
        idfs = {}
        D = len(self.crawled_data)
        # TODO remove this printf after verified that it is correct
        print("\nNumber of document in the corpus: ", D, "\ndelete this print pls\n")
        # Dts = {key: str, value: (Dt_value: int, seen_in_this_doc: bool)}
        Dts = {}
        for doc_id, doc_data in self.crawled_data.items(): # TODO .items()?
            tokens = doc_data.get("tokens")
            # TODO fix this check
            if tokens is None:
                continue
            # Reset visited values, since we are visiting a new document
            for key, (num, _) in Dts.items():
                Dts[key] = (num, False)

            for word in tokens:
                if Dts.get(word) is None:    # First time we see this term -> initialize it
                    Dts[word] = (1, True)
                elif Dts[word][1] is False:  # Dt already contains an entry for this term but is the first time we see it in this doc
                    Dts[word] = (Dts[word][0] + 1, True)   # Increment its frequency and toggle bool
                elif Dts[word][1] is True:
                    continue                # We've already seen this term in this doc
                else:
                    raise Exception('Something went wrong during building idfs.')
        
        for k, v in Dts.items():
            Dt = v[0]
            idfs[k] = math.log(D/Dt, 10)
            
        # Save data in file
        with open(self.path_to_IDFs, "wb") as f:
            pickle.dump(idfs, f)


    def _load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            if path == self.path_to_crawled_data:
                print("[INFO] No previous crawl data found, starting fresh.")
                return {}
            elif path == self.path_to_posting_lists:
                print("[INFO] No existing posting lists found, will build from scratch.")
                return {}, {}
            else:
                raise  # re-raise for unknown files


    def _can_skip(self, entry: list, other_doc: int) -> bool:
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


    def _intersect_skip(self, A: list, B: list) -> list:
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
                if _can_skip(A[i], B[j][0]):
                    i = A[i][1]
                    continue
                i += 1
            elif A[i][0] > B[j][0]:
                if _can_skip(B[j], A[i][0]):
                    j = B[j][1]
                    continue
                j += 1
            else:
                raise Exception("Something went wrong...")
        return matches


    def _in_range(self, A: list[int], B: list[int], rng: int) -> bool:
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


    def _intersect_range(self, A: list, B: list, rng: int) -> list:
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
                if _in_range(posA, posB, rng):
                    matches.append(docA)
                i += 1
                j += 1
            elif docA < docB:
                i += 1
            else:
                j += 1
        return matches