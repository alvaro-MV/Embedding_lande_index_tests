from embUtils import Chat
import embUtils as utils
from embRunIntra import run_expositive_mayeutic, run_intra
from embRun import run

def expositive():
	chunks = utils.separar_en_parrafos("data/fotosintesis.txt")
	lande_abstracts = run_intra(chunks, len(chunks))
	return (lande_abstracts)

def expositive_mayeutic():
	# chunks = utils.separar_en_parrafos("data/fotosintesis.txt")
	chunks =  [
        "Coffee is a brewed drink prepared from roasted coffee beans, the seeds of berries from certain Coffea species.",
        "From the coffee fruit, the seeds are separated to produce a stable, raw product: unroasted green coffee.",
        "The beans are roasted and then ground into a fine powder and brewed to make coffee.",
        "It is one of the most popular drinks in the world, and can be prepared and presented in a variety of ways."
    ]
	client = Chat()
	lande_expositive = run_expositive_mayeutic(chunks, client)
	return (lande_expositive)