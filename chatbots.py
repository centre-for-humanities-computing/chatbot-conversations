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
    training_args=None, verbose=False, use_files=True, use_gpu=-1, generation_parameters=None, context_size=600, full_conversation=True, output_path="outputs"):
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
        self.device = "cuda:" + str(use_gpu) if use_gpu > -1 else "cpu"
        
        self.population = population
        self.cycles = cycles
        self.initial_context = initial_context
        self.length = conversation_length
        self.output_path = output_path
        self.random_length = random_length
        self.trainer = trainer
        self.training_args = training_args
        self.verbose = verbose
        self.context_size = context_size
        self.use_files = use_files
        self.generation_parameters = generation_parameters
        self.full_conversation = full_conversation
        self.models = {}
        self.conversations = pd.DataFrame(columns=['Participant_1', 'Participant_2', 'Conversation', 'P_1_Conv', 'P_2_Conv'])

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
            print("Generating conversation between " + speaker_one + " and " + speaker_two)
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

            # if full context is required, use it, otherwise
            if self.context_size is not 0:
                input = c[-1 * self.context_size:]
            else:
                input = c

            input_ids = self.models[speakers[which_is_speaking]][0].encode(input, return_tensors='pt').to(self.device)

            if self.generation_parameters is not None:
                output = custom_generation(
                    self.models[speakers[which_is_speaking]][1].to(self.device),
                    self.device,
                    input_ids,
                    *self.generation_parameters
                )
            else:
                output = custom_generation(
                    self.models[speakers[which_is_speaking]][1].to(self.device),
                    self.device,
                    input_ids,
                    do_sample=True,
                    top_p = 0.95,
                    max_length=200*(lines+1),
                    length_penalty = 0.7
                )

            # return the decoded text + prep for next turn in conversation
            if self.context_size is not 0:
                c = c[:-1 * self.context_size] + self.models[speakers[which_is_speaking]][0].decode(output[0], skip_special_tokens=False) + "\n" + speakers[not which_is_speaking] + ":"
            else:
                c = self.models[speakers[which_is_speaking]][0].decode(output[0], skip_special_tokens=False) + "\n" + speakers[not which_is_speaking] + ":"
             
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
                        conv = self.conversation_generation(person, partner)
                        if self.use_files:
                            output = open(os.path.join(self.output_path, person + "_" + partner + "_" + str(datetime.datetime.now())),'w')
                            output.write(conv)
                            output.close()
                        else:
                            conv1, conv2 = self.split_conversation(conv, person, partner)
                            self.conversations.loc[len(self.conversations.index)]  = [person, partner, conv, conv1, conv2]

        # TODO: training here (need to decide how to train - maybe argument)
        #self.conversations.to_csv('conversations.txt', sep='\t', index=True)
        # get all of the output files into the conversation df to use for training
        if self.use_files:
            self.populate_conversations()
        self.train_participant("John")

    def train_participant(self, participant):
        if self.verbose:
            print("Training " + participant)
        context_length = 128
        tokenizer = self.models[participant][0]
        model = self.models[participant][1]

        # df implementation
        data = self.conversations[(self.conversations['Participant_1'] == participant) | (self.conversations['Participant_2'] == participant)]

        if self.full_conversation:
            conversations = data['Conversation'].tolist()
        else:
            spoke_first = data[data['Participant_1'] == participant]
            spoke_second = data[data['Participant_2'] == participant]
            conversations = spoke_first['P_2_Conv'].tolist() + spoke_second['P_1_Conv'].tolist()
        outputs = tokenizer(
        conversations,
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

    def return_model(self, participant):
        return self.models[participant][1]

    # read all of the output files and put them into the df in needed format
    # (only used if use_files is True)
    def populate_conversations(self):
        data_names = glob.glob(os.path.join(self.output_path, "*"))
        for file_name in data_names:
            with open(file_name, 'r') as f:
                person1, person2 = file_name.split('/')[1].split('_')[:2]
                conv = f.read()
                conv1, conv2 = self.split_conversation(conv, person1, person2)
                self.conversations.loc[len(self.conversations.index)]  = [person1, person2, conv, conv1, conv2]
            f.close()

    def split_conversation(self, conv, participant1, participant2):
        list_conv = conv.splitlines()
        conv1 = []
        conv2 = []
        for line in list_conv:
            if line.startswith(participant1):
                conv1.append(line)
            elif line.startswith(participant2):
                conv2.append(line)

        return "\n\n".join(conv1), "\n\n".join(conv2)

# TODO: test custom trainer, custom model + tokenizer, test custom parameters
# things left to do: custom parameters for generation, figure out training
# TODO: test split should be minimized (because the point is to train)