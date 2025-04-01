import pandas as pd
import embDataset
import embGet
import embUtils as utils
from embRun import run_on_expositive_texts
from embRunIntra import run_on_expositive_texts_intra


# Función para separar los abstract en chunnks, 
# que constituyen el texto incremental.
def split_into_chunks(text, n_chunks=8):
    text = str(text) if pd.notna(text) else ''
    chunk_len = len(text) // n_chunks
    return [text[i * chunk_len : (i + 1) * chunk_len] for i in range(n_chunks - 1)] + [text[(n_chunks - 1) * chunk_len:]]

def expositive(intra):
	if (intra == 1):
		chunks = utils.separar_en_parrafos("data/cat.txt")
		lande_abstracts = run_on_expositive_texts_intra(chunks, len(chunks))
	else:
		expositive = embDataset.KaggleDataset(
			"nechbamohammed/research-papers-dataset")
		expositive.load_dataset()
		expositive.get_df_from_dataset()
		random_sample = expositive.get_sample(10)
		random_sample = random_sample[['abstract', 'title']]
		random_sample['abstract_chunks'] = random_sample['abstract'].apply(lambda x: split_into_chunks(x, n_chunks=8))

		title_embeddings = embGet.get_embedding_sample(random_sample, 'title')
		abstract_chunks = random_sample['abstract_chunks'].iloc[1]
		# print(f"\ttitle: \n{title_embeddings}\n\tchunks: \n{abstract_chunks}\n")
		lande_abstracts = run_on_expositive_texts(abstract_chunks, title_embeddings, 8)
		print(f"lande abstract: {lande_abstracts}")
	return (lande_abstracts)
