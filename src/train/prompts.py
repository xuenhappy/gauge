def build_qa_prompt(example):
    return ('You are a question answering assistant. Answer the question based only on the provided context.\n\n'
        f"Context:\n{example['context']}\n\n"
        f"Question:\n{example['question']}\n\n"
        'Answer:\n')


def build_prompt_from_style(style: str):
    style = style.lower()
    if style != 'qa_standard':
        raise ValueError(style)
    return build_qa_prompt
