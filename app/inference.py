import torch
from config.settings import MAX_NEW_TOKENS, MAX_INPUT_TOKENS, TEMPERATURE
from app.model_loader import model_loader


def generate_response(prompt: str):
    model = model_loader.model
    tokenizer = model_loader.tokenizer

    # ─────────────────────────────────────────────
    # Chat messages
    # ─────────────────────────────────────────────
    messages = [
        {
            "role": "system",
            "content": (
                "You are a reasoning agent that explains solutions in clear, structured, step-by-step logic."
                "Break down problems into smaller parts, showing how each step leads to the next."
                "Ensure your reasoning is easy to follow, precise, and avoids unnecessary complexity."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # ─────────────────────────────────────────────
    # Apply chat template
    # ─────────────────────────────────────────────
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # ─────────────────────────────────────────────
    # Tokenize
    # ─────────────────────────────────────────────
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,            
        max_length=MAX_INPUT_TOKENS 
    ).to(model.device)

    # ─────────────────────────────────────────────
    # Generate
    # ─────────────────────────────────────────────
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            renormalize_logits=True
        )

    # ─────────────────────────────────────────────
    # Decode ONLY new tokens
    # ─────────────────────────────────────────────
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()
