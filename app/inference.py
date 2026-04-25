import torch
from config.settings import MAX_NEW_TOKENS, TEMPERATURE
from app.model_loader import model_loader


def generate_response(prompt: str):
    model = model_loader.model
    tokenizer = model_loader.tokenizer

    # ✅ STEP 1: Define chat roles
    messages = [
        {
            "role": "system",
            "content": (
                "You are a crypto market stabilizing agent operating in a DeFi liquidity pool. "
                "Your goal is to maintain price stability, prevent crashes, and manage volatility. "
                "You analyze market conditions such as price trends, liquidity, and volatility, "
                "and take strategic actions. "
                "Always respond in the format provided by the user."
           )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # ✅ STEP 2: Apply Qwen chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # ✅ STEP 3: Tokenize structured input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # ✅ STEP 4: Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3,   # 🔥 lower for stable replies
            top_p=0.9,
            do_sample=True
        )

    # ✅ STEP 5: Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ STEP 6: Clean output (remove prompt)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    return response