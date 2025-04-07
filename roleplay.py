import embDataset
from embDataset import HFDataset
import embUtils as utils
import embGet
from embRun import run_conversation

# Funcion que se le pasa a método
# trasnsform_dataset
def select_train_set(dataset):
	return dataset['train']

########## Llevar a cabo la conversación #############

# round_values = [5, 10, 20]
# role_batch_size = [3, 8, 15, 20]

round_values = [5]
role_batch_size = [3]

def run_experiment_1(roleplay : HFDataset, generics, k_values, conversation_rounds_values):
    results = []
    for k in k_values:
        roleplay_sample =roleplay.get_sample(k)
        print(roleplay_sample)
        label = roleplay_sample.iloc[0,:]['text']
        embeddings_el = embGet.get_embedding_sample(roleplay_sample, 'text')
        for rounds in conversation_rounds_values:
            lande_conversation, lande_intra, lande_baseline = run_conversation(embeddings_el, generics,
                                                                  label, rounds)

            result = {
                'k': k,
                'conversation_rounds': rounds,
                'lande_conversation': lande_conversation,
                'lande_intra' : lande_intra,
                'lande_baseline': lande_baseline
            }
            results.append(result)
            print(f"role_batch_size: {k}\t round: {rounds}")
    return results

 
##  Función que se llama desde el archivo main
def roleplay():
    roleplay = embDataset.HFDataset("hieunguyenminh/roleplay")
    roleplay.load_dataset()
    generics = embDataset.HFDataset("generics_kb")
    generics.load_dataset()
    roleplay.transform_dataset(select_train_set)
    generics.transform_dataset(select_train_set)

    roleplay.get_df_from_dataset()
    generics.get_df_from_dataset()
    resultado = run_experiment_1(roleplay, generics, role_batch_size, round_values)
    return resultado