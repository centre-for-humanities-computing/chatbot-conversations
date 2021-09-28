from transformers import AutoTokenizer, AutoModelForCausalLM

# cache stores the gpt-j model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache/huggingface/transformers")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache/huggingface/transformers")

# encode context the generation is conditioned on
"""input_ids = tokenizer.encode('Psychiatrist: Welcome back, Kevin. How have you been?\nKevin:', return_tensors='pt')

top_p_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=100,
    top_p = 0.92,
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(top_p_output[0], skip_special_tokens=True))"""

context = "Psychiatrist: Welcome back, Kevin. How have you been?\nKevin:"
for i in range(0, 5):
    input_ids = tokenizer.encode(context, return_tensors='pt')

    top_p_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=100 * (i + 1),
    top_p = 0.92
    )

    context = tokenizer.decode(top_p_output[0], skip_special_tokens=True)
    print("Output " + str(i) + ":\n" + 100 * '-')
    print(context)
    print("\n" + 100 * '-')
