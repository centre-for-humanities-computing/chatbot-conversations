from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Conversation
import os
import torch
import datetime

time0 = datetime.datetime.now()
cache_path = os.path.join(".cache", "huggingface", "transformers")

# cache stores the gpt-j model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_path)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_path)

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", cache_dir=cache_path)
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large", cache_dir=cache_path)

# set this so it doesn't get logged
model.config.pad_token_id = model.config.eos_token_id


# conv_pipeline = pipeline("conversational", model=model, tokenizer=tokenizer)
# resp_pipeline = pipeline("conversational", model=model, tokenizer=tokenizer)

# c = Conversation("Welcome back.")
# first = conv_pipeline(c)

# c2 = Conversation()

# for i in range(0, 5):
    # c2.add_user_input(first.generated_responses[-1])
    # second = conv_pipeline(c2)
    # c.add_user_input(second.generated_responses[-1])
    # first = conv_pipeline(c)
# print(first)

# speaker_one + ": " + context + " " + speaker_two + ":"
def conversation_generation(speaker_one, speaker_two, length, context):
    c = speaker_one + ": " + context + "\n\n" + speaker_two + ":"
    correct_lines = 0

    for i in range(0, length):
        input_ids = tokenizer.encode(c, return_tensors='pt')

        top_p_output = model.generate(
        input_ids, 
        do_sample=True, 
        max_length=100*(i+1),
        top_p = 0.95,
        length_penalty = 0.7
        )

        c = tokenizer.decode(top_p_output[0], skip_special_tokens=False)
        # print("Output " + str(i) + ":\n" + 100 * '-')
        # print(c)
        # print("\n" + 100 * '-')

        # remove lines which don't start with the correct participant
        # j % 4 == 0 and 
        lines = c.splitlines()
        for j in range(correct_lines, len(lines)):
            if  j % 2 == 0 and not (lines[j].startswith(speaker_two + ":") or
            lines[j].startswith(speaker_one + ":") or lines[j].endswith("says one final sentence before ending the conversation.")):
                break
            else:
                correct_lines += 1
        c = '\n'.join(lines[:correct_lines])

        if i == length - 3:
            endings = ['.', '?', '!']
            lines = c.splitlines()

            if not (lines[-1].endswith(tuple(endings)) or lines[-1] == ""):
                lines.pop()

            if lines[-1] != "":
                lines.append("")

            c = '\n'.join(lines)

            if (len(lines) - 2) % 4 == 0:
                c += '\n' + speaker_two + " says one final sentence before ending the conversation.\n\n" + speaker_two + ':'
            else:
                c += '\n' + speaker_one + " says one final sentence before ending the conversation.\n\n" + speaker_one + ':'



    # print(c)
    # print("Correct lines: ", len(lines))
    # input_ids = tokenizer.encode(c, return_tensors='pt')

    # print("FINISHED ENCODING")
    # top_p_output = model.generate(
    # input_ids, 
    # do_sample=True, 
    # max_length=len(c) + 100,
    # top_p = 0.93,
    # length_penalty = 0.7
    # )
    # c = tokenizer.decode(top_p_output[0], skip_special_tokens=False)

    return c

time1 = datetime.datetime.now()
# encode context the generation is conditioned on
conversation = conversation_generation("Psychiatrist", "Kevin", 6, "Welcome back, Kevin. How have you been?")
print("FINISHED GENERATING")
f = open(os.path.join("outputs", str(datetime.datetime.now())), 'w')
f.write(conversation)
f.write("Loading model: ", str(time1 - time0))
f.write("Generation: ", str(datetime.datetime.now() - time1))
f.close()
print("FINISHED EVERYTHING")