from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM, pipeline, Conversation, StoppingCriteriaList, MaxLengthCriteria
from datasets import load_dataset
import os
import torch
import datetime
import random
from generation import custom_generation

cache_path = os.path.join(".cache", "huggingface", "transformers")

class Experiment:
    def __init__(self, population, cycles, sample_strategy, initial_context, conversation_length):
        """a run of the experiment
        
        Keyword arguments:
        population -- the participants (and TODO: their models) of the experiment
        cycles -- one cycle is a conversation between every participant
        sample_strategy -- TBD
        initial_context -- how does participant_1 start the conversation?
        conversation_length -- how many lines should be spoken (TODO: maybe should not be a fixed number)
        Return: return_description
        """
        
        self.population = population
        self.cycles = cycles
        self.sample_strategy = sample_strategy
        self.initial_context = initial_context
        self.length = conversation_length

        # TODO: instantiate models here

    def conversation_generation(self, speaker_one, speaker_two, length, context, tokenizer, model):
        # TODO: stopping criteria
        # initial criteria
        c = "A conversation between " + speaker_one + " and " + speaker_two + ": \n\n" + speaker_one + ": " + context + "\n\n" + speaker_two + ":"
        correct_lines = 4

        # TODO: instead of for, should make a while loop (because of removing lines etc.)
        for i in range(length):
            input_ids = tokenizer.encode(c, return_tensors='pt')

            # top_p_output = model.sample(
            # input_ids, 
            # do_sample=True,
            # # TODO: make stopping_criteria work
            # # stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=100)),
            # # max_length=100*(i+1),
            # top_p = 0.95,
            # length_penalty = 0.7
            # )

            top_p_output = custom_generation(
                model,
                input_ids,
                do_sample=True,
                top_p = 0.95,
                max_length=200*(i+1),
                # length_penalty = 0.7
            )

            # print(top_p_output)
            # decode the contents and split it into lines for processing
            c = tokenizer.decode(top_p_output[0], skip_special_tokens=False)
            correct_lines_at_start = correct_lines
            lines = c.splitlines()
            print(i, c)

            # TODO: make sure this still works
            # if a line does not start with one of the speakers and is not blank, its incorrect and must not be used
            # TODO: check if the line actually ends
            for j in range(correct_lines, len(lines)):
                if  j % 2 == 0 and not (lines[j].startswith(speaker_two + ":") or
                lines[j].startswith(speaker_one + ":")):
                    break
                else:
                    correct_lines += 1
            # take only the correct lines, but also only take one side of the conversation
            c = '\n'.join(lines[:min(correct_lines, correct_lines_at_start+2)])

        return c


        
    def run(self):
        tokenizers = []
        models = []
        #for participant in self.population:

        # TODO: tokenizer/model for each person
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_path)

        # add the EOS token as PAD token to avoid warnings
        model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id, cache_dir=cache_path)

        for i in range(self.cycles):
            for person in self.population:
                for partner in self.population:
                    if partner is not person:
                        self.conversation_generation(person, partner, self.length, self.initial_context, tokenizer, model)



Experiment(
    population = {"John" : "gpt2", "Margaret" : "gpt2"}, 
    cycles = 1,
    # TODO: implement different strategies
    sample_strategy = "TBD", 
    #callbacks = [Metrics(), Print(), Logger()],
    # TODO: use_gpu
    #use_gpu = True
    initial_context="Hi. Why are you late?",
    conversation_length=3
    ).run()