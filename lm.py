import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from math import ceil

MODEL_NAME = "BanglaLLM/Bangla-s1k-qwen-2.5-3B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Use 4-bit quantization to reduce memory usage
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)
model.eval()


def fix_bangla_texts_batch(
    texts: list[str],
    batch_size: int = 1,
    max_new_tokens: int = 80
) -> list[str]:

    fixed_texts = []

    num_batches = ceil(len(texts) / batch_size)

    for i in range(num_batches):
        batch = texts[i * batch_size:(i + 1) * batch_size]

        # Process each text individually to reduce memory usage
        for text in batch:
            messages = [
                {
                    "role": "user",
                    "content": f"Fix and improve the following Bangla text:\n{text}"
                }
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            fixed_texts.append(decoded.strip())

            # Clear CUDA cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return fixed_texts


# Usage
cleaned_results = fix_bangla_texts_batch(results, batch_size=4)