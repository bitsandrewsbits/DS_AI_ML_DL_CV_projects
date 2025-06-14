# preprocessing Films Reviews Dataset files for BERT fine-tuning dataset creation.

class BERT_Fune_Tuning_Dataset_Creation:
    def __init__(self, dataset_vocabulary_file_path: str):
        self.dataset_vocab_file_path = dataset_vocabulary_file_path
        self.vocab_word_via_index_dict = {}

    def main(self):
        self.create_dict_for_access_vocabulary_word_via_index()

    def create_dict_for_access_vocabulary_word_via_index(self):
        with open(self.dataset_vocab_file_path, 'r') as dataset_vocab_file:
            for (i, word) in enumerate(dataset_vocab_file):
                self.vocab_word_via_index_dict[i] = word[:-1]
        return True

if __name__ == '__main__':
    dataset_files_preprocessing = BERT_Fune_Tuning_Dataset_Creation(
        'data/imdb.vocab'
    )
    dataset_files_preprocessing.main()
