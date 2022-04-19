from chatbots import Experiment
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import os

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained("gpt2",  cache_dir=os.path.join(".cache", "huggingface", "transformers"))

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

Experiment(
    #"EleutherAI/gpt-neo-2.7B"
    population = {"John" : "gpt2", "Margaret" : [tokenizer, model]}, 
    cycles = 2,
    initial_context="Hi. Why are you late?",
    conversation_length=10,
    verbose = True,
    use_gpu=0,
    use_files=False,
    full_conversation=False,
    training_args=args
    ).run()