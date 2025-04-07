import embDataset
import embUtils as utils
import embDataset
from embRun import Run, update_join_next_chunk, run_conversation

def expositive():
	chunks = utils.separar_en_parrafos("data/fotosintesis.txt")
	runner_expositive = Run(update_join_next_chunk, None, None, len(chunks))

	lande_abstracts = runner_expositive.run(None, chunks, 'intra')
	return (lande_abstracts)

def expositive_mayeutic():
	with open('data/coffe.txt', 'r', encoding='utf-8') as f:
		chunks = f.read()
	generics = embDataset.HFDataset("generics_kb")
	generics.load_dataset()
	generics.transform_dataset(embDataset.select_train_set)
	generics.get_df_from_dataset()
	lande_intra, lande_base = run_conversation(None, generics, chunks, 6)
	result = [{
		'k': 1,
		'conversation_rounds': 6,
		'lande_intra' : lande_intra,
		'lande_baseline': lande_base
	}]
	return (result)