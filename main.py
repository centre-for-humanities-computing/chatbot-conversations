from chatbots import Experiment

Experiment(
    #"EleutherAI/gpt-neo-2.7B"
    population = {"John" : "gpt2", "Margaret" : "gpt2"}, 
    cycles = 2,
    initial_context="Hi. Why are you late?",
    conversation_length=5,
    verbose = True,
    use_gpu=0,
    use_files=False,
    full_conversation=False
    ).run()