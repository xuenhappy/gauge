import yaml


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def align_model_and_tokenizer(model, tokenizer):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id