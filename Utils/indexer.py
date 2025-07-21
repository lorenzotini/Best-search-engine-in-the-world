from collections import defaultdict
import pickle
import math
from sentence_transformers import SentenceTransformer


class Indexer:
    def __init__(self, silent=False):
        self.path_to_TFs = 'data/tfs.pkl'
        self.path_to_IDFs = 'data/idfs.pkl'
        self.path_to_crawled_data = 'data/crawled_data.pkl'
        self.path_to_posting_lists = 'data/posting_list.pkl'
        self.path_to_embeddings='data/sbert_doc_embeddings.pkl'
        
        if not silent:
            self.crawled_data = self._load(self.path_to_crawled_data)
            
        self.skip_dict, self.pos_index_dict = self._load(self.path_to_posting_lists)


    def run(self):
        print('Indexing...')
        self._index_documents()
        print('Building Term frequencies...')
        self._build_TF()
        print('Building Inverse Document frequencies...')
        self._build_IDF()
        print("Precomputing doc embeddings.")
        self._precompute_document_embeddings()
        print("Indexer run done.")


    def _precompute_document_embeddings(self, model_name='all-MiniLM-L6-v2'):
        model = SentenceTransformer(model_name)

        doc_texts = []
        doc_ids = []

        for doc_id, doc_data in self.crawled_data.items():
            tokens = doc_data.get('tokens')
            if tokens is None:
                continue
            text = " ".join(tokens)
            doc_texts.append(text)
            doc_ids.append(doc_id)

        print(f"[INFO] Encoding {len(doc_texts)} documents with SBERT...")

        embeddings = model.encode(doc_texts, convert_to_numpy=True, batch_size=16, show_progress_bar=True)

        # Save as {doc_id: np.array}
        doc_embeddings = {doc_id: emb for doc_id, emb in zip(doc_ids, embeddings)}

        with open(self.path_to_embeddings, 'wb') as f:
            pickle.dump(doc_embeddings, f)

        print(f"[DONE] Saved to {self.path_to_embeddings}")
            

    def get_union_candidates(self, query):
        """
        Retrieve documents that contain at least one of the query terms.
        Args:
            query: list of str -> preprocessed query tokens

        Returns:
            set of int: Candidate document IDs
        """
        candidate_ids = set()
        for term in query:
            if term in self.skip_dict:
                postings = self.skip_dict[term]
                ids = [entry[0] for entry in postings]
                candidate_ids.update(ids)
        return list(candidate_ids)


    def proximity_bonus(self, query, doc_id, window=3):
        """
        Returns a bonus score if query terms occur within a certain window in the document.

        Args:
            query: list of str -> preprocessed query tokens
            doc_id: int -> document ID to check
            window: int -> maximum allowed distance between terms

        Returns:
            float: Bonus score (e.g., 1.0 for phrase match, 0.5 for close proximity, 0 otherwise)
        """
        # Collect positions of all query terms in the given doc
        positions_list = []
        for term in query:
            if term not in self.pos_index_dict:
                return 0  # If term doesn't exist in corpus, no bonus

            term_postings = self.pos_index_dict[term]
            doc_entry = next((entry for entry in term_postings if entry[0] == doc_id), None)
            if doc_entry is None:
                return 0  # Term not in this doc
            positions_list.append(doc_entry[1])  # Add list of positions

        # For phrase match (exact sequence):
        if self._is_exact_phrase(positions_list):
            return 1.0  # Full bonus

        # For proximity match (within window)
        if self._in_proximity(positions_list, window):
            return 0.5  # Partial bonus

        return 0

    def _is_exact_phrase(self, positions_list):
        """
        Check if terms occur as an exact phrase in order.

        Args:
            positions_list: list of lists -> positions of each term in doc

        Returns:
            bool: True if terms appear consecutively in order
        """
        # Start with positions of the first term
        first_term_positions = positions_list[0]

        for pos in first_term_positions:
            match = True
            current_pos = pos
            for i in range(1, len(positions_list)):
                # Check if next term occurs at current_pos + 1
                if (current_pos + 1) in positions_list[i]:
                    current_pos += 1
                else:
                    match = False
                    break
            if match:
                return True  # Found exact phrase
        return False


    def _in_proximity(self, positions_list, window):
        """
        Check if terms occur close to each other (order doesn't matter).

        Args:
            positions_list: list of lists -> positions of each term in doc
            window: int -> max distance between farthest and closest term

        Returns:
            bool: True if terms are within the window somewhere in the doc
        """
        import itertools

        # Create all combinations of one position per term
        for combination in itertools.product(*positions_list):
            if max(combination) - min(combination) <= window:
                return True
        return False

    def _index_documents(self):
        self._build_skip_pointers(self.crawled_data)
        self._build_positional_index(self.crawled_data)
        
        with open(self.path_to_posting_lists, "wb") as f:
            pickle.dump((self.skip_dict, self.pos_index_dict), f)


    def _build_skip_pointers(self, crawled_data):
        token_to_ids = {}
        skip_dict = {}

        # Process every document
        for doc_id, doc_data in crawled_data.items():
            tokens = doc_data.get("tokens")
            
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
        # Dts = {key: str, value: (Dt_value: int, seen_in_this_doc: bool)}
        Dts = {}
        for doc_id, doc_data in self.crawled_data.items():
            tokens = doc_data.get("tokens")
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
                if self._can_skip(A[i], B[j][0]):
                    i = A[i][1]
                    continue
                i += 1
            elif A[i][0] > B[j][0]:
                if self._can_skip(B[j], A[i][0]):
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
                if self._in_range(posA, posB, rng):
                    matches.append(docA)
                i += 1
                j += 1
            elif docA < docB:
                i += 1
            else:
                j += 1
        return matches