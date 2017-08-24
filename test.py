from spacy.vocab import write_binary_vectors
import spacy.en
from os import path

def main(bz2_loc, bin_loc=None):
        if bin_loc is None:
                bin_loc = path.join(path.dirname(spacy.en.__file__), 'data', 'vocab', 'vec.bin')
        print bin_loc
	write_binary_vectors(bz2_loc, bin_loc)

main("")
