from chatting_chatbots import Experiment

Experiment(
    population = {"John" : "gpt2", "Margaret" : "gpt2", "Alice" : "gpt2", "Bob" : "gpt2", "Eve" : "gpt2", "Charlie" : "gpt2", "Darwin" : "gpt2"}, 
    cycles = 2,
    initial_context="Hi. Why are you late?",
    conversation_length=10,
    verbose = True,
    use_gpu=0,
    use_files=False
    ).run()