from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM, pipeline, Conversation, StoppingCriteriaList, MaxLengthCriteria, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import os
import torch
import datetime
import random
from generation import custom_generation
from datasets import load_dataset
from utils import get_participant_files
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

cache_path = os.path.join(".cache", "huggingface", "transformers")

class Experiment:
    def __init__(self, population, cycles=1, initial_context="Hello.", conversation_length=10, random_length=0, trainer=None,
    training_args=None, verbose=False, output_path="outputs"):
        """a run of the experiment
        
        Keyword arguments:
        population -- the participants (and their models) of the experiment
        cycles -- one cycle is a conversation between every participant
        sample_strategy -- TBD
        initial_context -- how does participant_1 start the conversation?
        conversation_length -- how many lines should be spoken (ignoring initial context)
        random_length -- if set to some value, sets the conversation length to be a random value from [c_l - r_l, c_l + r_l]
        output_path -- path for conversation outputs (default=create directory "outputs")
        Return: return_description
        """
        # TODO: should the text be saved to a file? make it into an argument

        # TODO: use_gpu
        #use_gpu = True
        
        self.population = population
        self.cycles = cycles
        self.initial_context = initial_context
        self.length = conversation_length
        self.output_path = output_path
        self.random_length = random_length
        self.trainer = trainer
        self.training_args = training_args
        self.verbose = verbose
        self.models = {}
        # TODO: think about whether we need more columns in the df
        self.conversations = pd.DataFrame(columns=['Participant_1', 'Participant_2', 'Conversation'])

        for participant in self.population:
            # if text is passed, assume it is a huggingface model
            if isinstance(self.population[participant], str):
                tokenizer = AutoTokenizer.from_pretrained(self.population[participant], cache_dir=cache_path)
                # add the EOS token as PAD token to avoid warnings
                model = AutoModelForCausalLM.from_pretrained(self.population[participant], pad_token_id=tokenizer.eos_token_id, cache_dir=cache_path)
                self.models[participant] = [tokenizer, model]

            # otherwise the tokenizer and model need to be passed
            else:
                self.models[participant] = self.population[participant]

            if self.verbose:
                print("Finished model instantiation")
        
    def conversation_generation(self, speaker_one, speaker_two):

        if self.verbose:
            print("Generating conversation between" + speaker_one + " and " + speaker_two)
        # initial criteria
        c = "A conversation between " + speaker_one + " and " + speaker_two + ": \n\n" + speaker_one + ": " + self.initial_context + "\n\n" + speaker_two + ":"
        lines = 0

        # boolean of which participant is speaking
        # 0: 1st participant, 1: 2nd participant
        which_is_speaking = 1
        speakers = [speaker_one, speaker_two]

        # get some randomness in length
        length = self.length + random.randint(-1 * self.random_length, self.random_length)

        # generate as many lines as set
        while lines < length:

            # encode input, generate output and decode it
            # TODO: allow option to either generate with full context or with last x chars
            input_ids = self.models[speakers[which_is_speaking]][0].encode(c[-600:], return_tensors='pt')
            top_p_output = custom_generation(
                self.models[speakers[which_is_speaking]][1],
                input_ids,
                # TODO: allow custom parameters
                do_sample=True,
                top_p = 0.95,
                max_length=200*(lines+1),
                length_penalty = 0.7
            )
            # TODO: dont forget above TODO here
            c = c[:-600] + self.models[speakers[which_is_speaking]][0].decode(top_p_output[0], skip_special_tokens=False) + "\n" + speakers[not which_is_speaking] + ":"
             
            # while loop stuff
            lines += 1
            which_is_speaking = not which_is_speaking

        return c


        
    def run(self):
        # each cycle is each participant speaking with each other participant
        # (from both sides)
        for i in range(self.cycles):
            for person in self.population:
                for partner in self.population:
                    if partner is not person:
                        self.conversations.loc[len(self.conversations.index)]  = [person, partner, self.conversation_generation(person, partner)]
                        # TODO: need an argument (whether to save to file or just df)
                        # output = open(os.path.join(self.output_path, person + "_" + partner + "_" + str(datetime.datetime.now())),'w')
                        # output.write(self.conversation_generation(person, partner))
                        # output.close()
        # TODO: training here (need to decide how to train - maybe argument)
        self.conversations.to_csv('conversations.txt', sep='\t', index=True)
        self.train_participant("John")

    def train_participant(self, participant):
        if self.verbose:
            print("Training " + participant)
        context_length = 128
        # TODO: this should be different depending on whether its df or file based
        # get all the output files where the person participated
        #data_names = get_participant_files(self.output_path, participant)
        tokenizer = self.models[participant][0]
        model = self.models[participant][1]

        # df implementation
        data = self.conversations[(self.conversations['Participant_1'] == participant) | (self.conversations['Participant_2'] == participant)]

        outputs = tokenizer(
        data['Conversation'].tolist(),
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
        )

        if self.verbose:
            print(f"Input IDs length: {len(outputs['input_ids'])}")
            print(f"Input chunk lengths: {(outputs['length'])}")
            print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

        # these are the batches of tokens I will use for training
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        
        tokenized_train, tokenized_test = train_test_split(input_batch, shuffle=False)

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        if self.training_args is None:
            args = TrainingArguments(
            output_dir="codeparrot-ds",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy="steps",
            eval_steps=5_000,
            logging_steps=5_000,
            gradient_accumulation_steps=8,
            num_train_epochs=5,
            weight_decay=0.1,
            warmup_steps=1_000,
            lr_scheduler_type="cosine",
            learning_rate=5e-4,
            save_steps=5_000,
            fp16=True,
            push_to_hub=False,
            )
        else:
            args = self.training_args

        if self.trainer is None:
            Trainer(
                model=model,
                tokenizer=tokenizer,
                args=args,
                data_collator=data_collator,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
            ).train()
        else:
            self.trainer(model=model,
                tokenizer=tokenizer,
                args=args,
                data_collator=data_collator,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
            ).train()
       # TODO: will train on the full file at the moment, but could be changed to train on just the new text (generated by other person)

# TODO: test custom trainer, custom model + tokenizer