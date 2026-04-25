import torch
from config.settings import MAX_NEW_TOKENS, TEMPERATURE
from app.model_loader import model_loader


def generate_response(prompt: str):
    model = model_loader.model
    tokenizer = model_loader.tokenizer

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)