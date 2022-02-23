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
        population -- the participants (and their models) of the experiment
        cycles -- one cycle is a conversation between every participant
        sample_strategy -- TBD
        initial_context -- how does participant_1 start the conversation?
        conversation_length -- how many lines should be spoken (ignoring initial context) (TODO: maybe should not be a fixed number)
        Return: return_description
        """
        
        self.population = population
        self.cycles = cycles
        self.sample_strategy = sample_strategy
        self.initial_context = initial_context
        self.length = conversation_length
        self.models = {}

        for participant in self.population:
            tokenizer = AutoTokenizer.from_pretrained(self.population[participant], cache_dir=cache_path)

            # add the EOS token as PAD token to avoid warnings
            model = AutoModelForCausalLM.from_pretrained(self.population[participant], pad_token_id=tokenizer.eos_token_id, cache_dir=cache_path)
            # TODO: enum for tokenizer, model
            self.models[participant] = [tokenizer, model]
        
        # instantiating 2 identical models might use the same model underneath (not sure - maybe not?)

    def conversation_generation(self, speaker_one, speaker_two):

        # initial criteria
        c = "A conversation between " + speaker_one + " and " + speaker_two + ": \n\n" + speaker_one + ": " + self.initial_context + "\n\n" + speaker_two + ":"
        lines = 0

        # boolean of which participant is speaking
        # 0: 1st participant, 1: 2nd participant
        which_is_speaking = 1
        speakers = [speaker_one, speaker_two]

        while lines < self.length:

            input_ids = self.models[speakers[which_is_speaking]][0].encode(c, return_tensors='pt')

            top_p_output = custom_generation(
                self.models[speakers[which_is_speaking]][1],
                input_ids,
                # TODO: allow custom parameters
                do_sample=True,
                top_p = 0.95,
                max_length=200*(lines+1),
                length_penalty = 0.7
            )

            # decode the contents and split it into lines for processing
            c = self.models[speakers[which_is_speaking]][0].decode(top_p_output[0], skip_special_tokens=False)
            # correct_lines_at_start = correct_lines
            # lines = c.splitlines()
            print(lines, c)

            # TODO: make sure this still works
            # if a line does not start with one of the speakers and is not blank, its incorrect and must not be used
            # TODO: check if the line actually ends
            # for j in range(correct_lines, len(lines)):
            #     if  j % 2 == 0 and not (lines[j].startswith(speaker_two + ":") or
            #     lines[j].startswith(speaker_one + ":")):
            #         break
            #     else:
            #         correct_lines += 1
            # # take only the correct lines, but also only take one side of the conversation
            # c = '\n'.join(lines[:min(correct_lines, correct_lines_at_start+2)])

            # while loop stuff
            lines += 1
            which_is_speaking = not which_is_speaking

        return c


        
    def run(self):
        for i in range(self.cycles):
            for person in self.population:
                for partner in self.population:
                    if partner is not person:
                        self.conversation_generation(person, partner)



Experiment(
    population = {"John" : "gpt2", "Margaret" : "EleutherAI/gpt-neo-2.7B"}, 
    cycles = 1,
    # TODO: implement different strategies
    sample_strategy = "TBD", 
    #callbacks = [Metrics(), Print(), Logger()],
    # TODO: use_gpu
    #use_gpu = True
    initial_context="Hi. Why are you late?",
    conversation_length=3
    ).run()