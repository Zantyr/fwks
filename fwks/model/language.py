import pynini
import tqdm


class Alphabet:
    """
    Represents mapping between phonemes in IPA
    and representations in text files
    """
    def __init__(self):
        self.symbols
        

class Vocabulary:
    """
    Get ...
    """
        
    
class Allophony:
    """
    Multiplies the vocabulary by taking into account all possible contexts
    of words (endings may be voiced e.g.)
    """
    



class LanguageModel:
    def __init__(self, vocab=None):
        self.vocab = vocab
        self.coefficients = [10]  # To Be Implemented'
        self.basic_occurrences_of_gram = 1
        # https://www.isip.piconepress.com/courses/msstate/ece_8463/lectures/current/lecture_33/lecture_33.pdf - n-gram smoothing

    def build(self, dataset):
        # get vocabulary if needed
        # get words from an adapter
        # get transcript words
        self.chain = pynini.Fst()
        self.word_symbols = {}
        for word in transcript_words:
            state = self.chain.add_symbol()
        print("Getting word transcriptions")
        for sentence in tqdm.tqdm(transcript):
            for word in sentence:
                for word_trans in word:
                    start_state = ...
                    for phon in word_trans:
                        # add arcs per each phoneme
                        self.chain.add_arc(Arc())  
                    # add epsilons to the dummy state for new word
        print("Optimizing the model")
        # minimize()
        self.chain.determinize()

    @classmethod
    def load(self, path):
        pass

    def join_fst(self, acoustic_fst):
        return pynini.compose(acoustic_fst, self.language_fst)
    
    def sentence_hypotheses(self, acoustic_fst):
        sentence = self.join_fst(acoustic_fst)
        sentence = pynini.pdt_shortestpath(sentence)
        return [self.vocab[x] for x in sentence]

