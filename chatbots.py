from transformers import AutoTokenizer, AutoModelForCausalLM

# cache stores the gpt-j model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache/huggingface/transformers")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=".cache/huggingface/transformers")


def conversation_generation(speaker_one, speaker_two, length, context):
    c = speaker_one + ": " + context + "\n" + speaker_two + ":"

    for i in range(0, length):
        input_ids = tokenizer.encode(c, return_tensors='pt')

        top_p_output = model.generate(
        input_ids, 
        do_sample=True, 
        max_length=100*(i+1),
        top_p = 0.92
        )

        c = tokenizer.decode(top_p_output[0], skip_special_tokens=False)
        print("Output " + str(i) + ":\n" + 100 * '-')
        print(c)
        print("\n" + 100 * '-')


# encode context the generation is conditioned on
conversation_generation("Psychiatrist", "Kevin", 5, "Welcome back, Kevin. How have you been?")