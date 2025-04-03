import pandas as pd
import embDataset
import embGet
import embUtils as utils
from embRunIntra import run_on_expositive_texts_intra

def expositive():
	chunks = utils.separar_en_parrafos("data/fotosintesis.txt")
	lande_abstracts = run_on_expositive_texts_intra(chunks, len(chunks))
	return (lande_abstracts)
