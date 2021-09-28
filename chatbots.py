from transformers import AutoTokenizer, AutoModelForCausalLM

# cache stores the gpt-j model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache/huggingface/transformers")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache/huggingface/transformers")

# encode context the generation is conditioned on
input_ids = tokenizer.encode('Psychiatrist: Welcome back, Kevin. How have you been?\nKevin:', return_tensors='pt')

top_k_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=100, 
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(top_k_output[0], skip_special_tokens=True))
