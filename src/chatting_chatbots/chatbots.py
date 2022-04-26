from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import os
import torch
import datetime
import random
from .generation import custom_generation
from .utils import get_participant_files, split_conversation
import glob
import pandas as pd


class Experiment:
    def __init__(
        self,
        population,
        cycles=1,
        initial_context="Hello.",
        conversation_length=10,
        random_length=0,
        training_args=None,
        verbose=False,
        use_files=True,
        use_gpu=-1,
        generation_parameters=None,
        context_size=600,
        full_conversation=True,
        batch_size=128,
        train_after_run=True,
        output_path="outputs",
        cache_path=os.path.join(".cache", "huggingface", "transformers"),
    ):
        """an initialization of the experiment

        Keyword arguments:
        population -- the participants (and their models) of the experiment
        cycles -- one cycle is a conversation between every participant
        initial_context -- how does participant_1 start the conversation?
        conversation_length -- how many lines should be spoken (ignoring initial context)
        random_length -- if set to some value, sets the conversation length to be a random value from [c_l - r_l, c_l + r_l]
        training_args -- custom arguments to Trainer function
        verbose -- adds some text to be printed if set to True
        use_files -- whether files should be used for training (or just the conversations generated during a run)
        use_gpu -- uses cpu if set to -1, uses the according gpu if set to some number > -1
        generation_parameters -- custom parameters for generating text (not tested)
        context_size -- how many tokens should be used to generate the text. set to 0 to use all of the tokens (crashes on big contexts)
        full_conversation -- if set to False, will only train on the other person's text
        batch_size -- size of batches for training
        train_after_run -- if set to True, trains each participant after a run
        output_path -- path for conversation outputs
        cache_path -- path for storing the models that are downloaded
        """
        self.device = "cuda:" + str(use_gpu) if use_gpu > -1 else "cpu"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.population = population
        self.cycles = cycles
        self.initial_context = initial_context
        self.length = conversation_length
        self.output_path = output_path
        self.random_length = random_length
        self.training_args = training_args
        self.verbose = verbose
        self.context_size = context_size
        self.use_files = use_files
        self.generation_parameters = generation_parameters
        self.full_conversation = full_conversation
        self.batch_size = batch_size
        self.train_after_run = train_after_run
        self.models = {}
        self.conversations = pd.DataFrame(
            columns=[
                "Participant_1",
                "Participant_2",
                "Conversation",
                "P_1_Conv",
                "P_2_Conv",
            ]
        )

        for participant in self.population:
            # if text is passed, assume it is a huggingface model
            if isinstance(self.population[participant], str):
                tokenizer = AutoTokenizer.from_pretrained(
                    self.population[participant], cache_dir=cache_path
                )
                # add the EOS token as PAD token to avoid warnings
                model = AutoModelForCausalLM.from_pretrained(
                    self.population[participant],
                    pad_token_id=tokenizer.eos_token_id,
                    cache_dir=cache_path,
                ).to(self.device)
                self.models[participant] = [tokenizer, model]
            # otherwise the tokenizer and model need to be passed
            else:
                self.models[participant] = [0, 0]
                self.models[participant][0] = self.population[participant][0]
                self.models[participant][1] = self.population[participant][1].to(
                    self.device
                )
                self.models[participant][1].config.pad_token_id = self.models[
                    participant
                ][0].eos_token_id

        if self.verbose:
            print("Finished model instantiation")

    def conversation_generation(self, speaker_one, speaker_two):
        """generate a conversation between two participants

        Keyword arguments:
        speaker_one - name of speaker one (as a string; must exist in the participants pool)
        speaker_two - name of speaker two (rules as above)
        Return: a conversation between the two speakers
        """

        if self.verbose:
            print(
                "Generating conversation between " + speaker_one + " and " + speaker_two
            )

        # initial criteria
        c = (
            "A conversation between "
            + speaker_one
            + " and "
            + speaker_two
            + ": \n\n"
            + speaker_one
            + ": "
            + self.initial_context
            + "\n\n"
            + speaker_two
            + ":"
        )
        lines = 0

        # boolean of which participant is speaking
        # 0: 1st participant, 1: 2nd participant
        which_is_speaking = 1
        speakers = [speaker_one, speaker_two]

        # get some randomness in length
        length = self.length + random.randint(
            -1 * self.random_length, self.random_length
        )

        # generate as many lines as set
        while lines < length:
            # if full context is required, use it, otherwise use the last x tokens
            if self.context_size is not 0:
                input = c[-1 * self.context_size :]
            else:
                input = c

            # encode the context
            input_ids = (
                self.models[speakers[which_is_speaking]][0]
                .encode(input, return_tensors="pt")
                .to(self.device)
            )

            # generate the next line; custom generation generates text until \n is generated, where it breaks
            # in this way, the model generating the text can be changed on each turn
            if self.generation_parameters is not None:
                output = custom_generation(
                    self.models[speakers[which_is_speaking]][1],
                    self.device,
                    input_ids,
                    *self.generation_parameters,
                )
            else:
                output = custom_generation(
                    self.models[speakers[which_is_speaking]][1],
                    self.device,
                    input_ids,
                    do_sample=True,
                    top_p=0.95,
                    max_length=200 * (lines + 1),
                    length_penalty=0.7,
                )

            # return the decoded text + prep for next turn in conversation
            if self.context_size is not 0:
                c = (
                    c[: -1 * self.context_size]
                    + self.models[speakers[which_is_speaking]][0].decode(
                        output[0], skip_special_tokens=False
                    )
                    + "\n"
                    + speakers[not which_is_speaking]
                    + ":"
                )
            else:
                c = (
                    self.models[speakers[which_is_speaking]][0].decode(
                        output[0], skip_special_tokens=False
                    )
                    + "\n"
                    + speakers[not which_is_speaking]
                    + ":"
                )

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
                        # generate conversation, save to files if needed, save to df otherwise
                        conv = self.conversation_generation(person, partner)
                        if self.use_files:
                            output = open(
                                os.path.join(
                                    self.output_path,
                                    person
                                    + "_"
                                    + partner
                                    + "_"
                                    + str(datetime.datetime.now()),
                                ),
                                "w",
                            )
                            output.write(conv)
                            output.close()
                        else:
                            conv1, conv2 = split_conversation(conv, person, partner)
                            self.conversations.loc[len(self.conversations.index)] = [
                                person,
                                partner,
                                conv,
                                conv1,
                                conv2,
                            ]

        # get all of the output files into the conversation df to use for training
        if self.use_files:
            self.populate_conversations()
        # train each participant
        if self.train_after_run:
            for person in self.population:
                self.train_participant(person)

    def train_participant(self, participant):
        """function for training one of the participants

        Keyword arguments:
        participant - name of participant (in str form, must exist in Experiment pool)
        """

        if self.verbose:
            print("Training " + participant)

        # only take conversations where the participant was talking
        data = self.conversations[
            (self.conversations["Participant_1"] == participant)
            | (self.conversations["Participant_2"] == participant)
        ]

        # use either all of the conversation or the partners text
        if self.full_conversation:
            conversations = data["Conversation"].tolist()
        else:
            spoke_first = data[data["Participant_1"] == participant]
            spoke_second = data[data["Participant_2"] == participant]
            conversations = (
                spoke_first["P_2_Conv"].tolist() + spoke_second["P_1_Conv"].tolist()
            )

        # tokenize the text used for training
        outputs = self.models[participant][0](
            conversations,
            truncation=True,
            max_length=self.batch_size,
            return_overflowing_tokens=True,
            return_length=True,
        )

        if self.verbose:
            print(f"Input IDs length: {len(outputs['input_ids'])}")
            print(f"Input chunk lengths: {(outputs['length'])}")
            print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

        # only use inputs with required length
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == self.batch_size:
                input_batch.append(input_ids)

        data_collator = DataCollatorForLanguageModeling(
            self.models[participant][0], mlm=False
        )

        # some default args if custom ones are not given
        if self.training_args is None:
            args = TrainingArguments(
                output_dir="training-dir",
                per_device_train_batch_size=2,
                evaluation_strategy="steps",
                logging_steps=500,
                gradient_accumulation_steps=2,
                num_train_epochs=1,
                weight_decay=0.1,
                warmup_steps=500,
                lr_scheduler_type="cosine",
                learning_rate=5e-4,
                save_steps=500,
                fp16=True,
                push_to_hub=False,
            )
        else:
            args = self.training_args

        # train the model
        Trainer(
            model=self.models[participant][1],
            tokenizer=self.models[participant][0],
            args=args,
            data_collator=data_collator,
            train_dataset=input_batch,
        ).train()

    # helper function for returing the model of a participant
    def return_model(self, participant):
        return self.models[participant][1]

    # read all of the output files and put them into the df in needed format
    # (only used if use_files is True)
    def populate_conversations(self):
        data_names = glob.glob(os.path.join(self.output_path, "*"))
        self.conversations = pd.DataFrame(
            columns=[
                "Participant_1",
                "Participant_2",
                "Conversation",
                "P_1_Conv",
                "P_2_Conv",
            ]
        )

        for file_name in data_names:
            with open(file_name, "r") as f:
                person1, person2 = file_name.split("/")[1].split("_")[:2]
                conv = f.read()
                conv1, conv2 = split_conversation(conv, person1, person2)
                self.conversations.loc[len(self.conversations.index)] = [
                    person1,
                    person2,
                    conv,
                    conv1,
                    conv2,
                ]
            f.close()
