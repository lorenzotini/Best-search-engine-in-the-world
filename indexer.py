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

"""     def _load(self, path):
        if not os.path.exists(path):
            print("[INFO] No previous crawl data found, starting fresh.")
            return {}
        if os.path.getsize(path) == 0:
            print("[INFO] Found empty crawl data file, starting fresh.")
            return {}
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data """
    

        
