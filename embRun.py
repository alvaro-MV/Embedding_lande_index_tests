import torch
import embUtils


def run_lande_loop(
    embedding_reference,
    n_steps,
    update_fn,
    input_fn,
    accumulate=True,
    compute_baseline_fn=None
):
    accumulated = ""
    lande_values = []
    baseline_values = []

    for step in range(n_steps):
        new_piece = input_fn(step, accumulated)
        if accumulate:
            accumulated += new_piece + "|"
        else:
            accumulated = new_piece  # useful if you want to only use the current piece

        t_emb = torch.Tensor(generate_embeddings(accumulated)).unsqueeze(0)
        lv = lande_index(t_emb, embedding_reference)
        lande_values.append(lv)

        if compute_baseline_fn is not None:
            baseline = compute_baseline_fn()
            t_base = torch.Tensor(generate_embeddings(baseline))
            lb = lande_index(t_base, embedding_reference)
            baseline_values.append(lb)

    return lande_values, baseline_values if compute_baseline_fn else lande_values

def run_conversation(embeddings_el, generics_df, label, conversation_rounds=10):
    def input_fn(step, accumulated):
        question = get_questions(accumulated, roleplay_task_indication)
        answer = correct_questions(question, label)
        return answer

    def baseline_fn():
        return extractBaselineAnswer(generics_df)

    return run_lande_loop(
        embeddings_el,
        conversation_rounds,
        update_fn=None,  # included in input_fn
        input_fn=input_fn,
        compute_baseline_fn=baseline_fn
    )

def run_on_expositive_texts(abstract_chunks, title_embs, n_batches=8):
    def input_fn(step, accumulated):
        return abstract_chunks[step]

    return run_lande_loop(
        title_embs,
        n_batches,
        update_fn=None,
        input_fn=input_fn
    )





# def run_conversation(embeddings_el, generics_df, label, conversation_rounds = 10):
#   Answers = ""
#   lande_conversation = []
#   lande_baseline = []
#   print(conversation_rounds)
#   while conversation_rounds > 0:
#       question = get_questions(Answers, roleplay_task_indication)
#       answer = correct_questions(question, label)

#       Answers += "|" + answer
#       #print(f"Answers: -----------: {Answers}\n")
#       t_answer = torch.Tensor(generate_embeddings(Answers)).unsqueeze(0)
#       t_base = extractBaselineAnswer(generics_df)
#       t_base = torch.Tensor(generate_embeddings(t_base))
#       la = lande_index(t_answer, embeddings_el)
#       lb = lande_index(t_base, embeddings_el)
#       #print(f"li: {la}\n")
#       lande_conversation.append(la)
#       lande_baseline.append(lb)
#       conversation_rounds = conversation_rounds - 1
#   return lande_conversation, lande_baseline


# def run_on_expositive_texts(abstract_chunks, title_embs, n_batches = 8):
#   Abstract = ""
#   i = 0
#   lande_abstract = []
#   while i < n_batches:
#       Abstract += abstract_chunks[i] + "|"
#       #print(f"Answers: -----------: {Answers}\n")
#       t_chunks = torch.Tensor(generate_embeddings(Abstract)).unsqueeze(0)
#       la = lande_index(t_chunks, title_embs)
#       #print(f"li: {la}\n")
#       lande_abstract.append(la)
#       i += 1
#   return lande_abstract