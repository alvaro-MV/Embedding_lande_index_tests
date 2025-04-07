from embUtils import Chat
import embUtils as utils
from embRun import Run, update_join_next_chunk, update_mayeutic_answer

def expositive():
	chunks = utils.separar_en_parrafos("data/fotosintesis.txt")
	runner_expositive = Run(update_join_next_chunk, None, None, len(chunks))

	lande_abstracts = runner_expositive.run(None, chunks, 'intra')
	return (lande_abstracts)

def expositive_mayeutic():
	chunks = utils.separar_en_parrafos("data/fotosintesis.txt")
	client = Chat()
	runner_expositive = Run(update_mayeutic_answer, None, client, len(chunks))
	lande_expositive = runner_expositive.run(None, chunks, 'intra')
	return (lande_expositive)