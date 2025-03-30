import openai
import random
from torch.nn import functional as F

def lande_index(label, el):
    vector = F.cosine_similarity(label, el)
    soft_vector = F.softmax(vector, 0)
    print(f'the soft vector is: {soft_vector}\n')
    N = len(soft_vector)
    s_div = 0
    for i in range(0, N-1):
        s_div += (soft_vector[i]) ** 2
    s_div = 1 - s_div
    return (N/(N-1))*s_div

def chatgpt__call__(client, prompt, model="gpt-4-turbo"):
    messages = [{'role': "user", "content": prompt}]
    completition = client.chat.completions.create(
    model=model,
    messages=messages
    )
    return completition.choices[0].message.content

def get_questions(history, task_indication, num_questions = 4):

        questions_prompt = f"""
  You are an Artificial Intelligence (AI) agent. Imagine that you have access to a text,
    showing a conversation. Yo will be provided with the history of questions and the corrections propousal to those questions.
        In addition, the initial indication about the text is {task_indication}.
        Your purpouse is to ask questions about the conversation.
        The previous corrections where {history}.

        Your task is to improve the previous questions to reach a lower metric
        (closer approximation to the object). Generate {num_questions} questions.
        Separate the questions by end of line ('\n') character.

        Questions:
        """

        generated_questions = chatgpt__call__(questions_prompt)
        return generated_questions

def correct_questions(questions, context):

	correction_prompt = f"""
	You are an Artificial Intelligence (AI) agent. Your purpose is to answer questions posed to you.
	You must extract the answers from the text presented below and elaborate a synthesis.
	You should focus only on the information provided. The description is as follows: {context}.
	The questions to answer are {questions}. Each question is separated by \n character.

	IMPORTANT!!
	Do NOT by any chance mention the name of any of the characters that appear in the description.
	You must propose as well a new question that could contain more epistemological knowledge about the major subject.
	Answer:
	New question:
	"""

	answer = chatgpt__call__(correction_prompt)

	return answer

# Funcion para extraer una frase
# aleatoria que conformará la línea base
# del índice de Lande.
def extractBaselineAnswer(df_base):
  random_element = df_base.iloc[random.choices(range(0, df_base.index.stop -1), k=1)]
  return list(random_element.generic_sentence.values)[0]