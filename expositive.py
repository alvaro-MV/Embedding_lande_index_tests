import embDataset
import embUtils as utils
import embDataset
import os
from embRun import Run, update_join_next_chunk, run_conversation

def expositive():
	result = []
	# files = os.listdir("data/texts")
	# for file in files:
	chunks = utils.separar_en_parrafos("data/texts/philo.txt")
	runner_expositive = Run(update_join_next_chunk, None, None, len(chunks))
	lande_expositive, i = runner_expositive.run(None, chunks, 'intra')
	print(f"lande_expositive: {lande_expositive}\n")
	
	result.append({
		'text_name' : "philo",
		'n_paragraphs' : i,
		'lande_intra' : lande_expositive
	})
	return (result)

def expositive_mayeutic():
	with open('data/texts/cognition.txt', 'r', encoding='utf-8') as f:
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