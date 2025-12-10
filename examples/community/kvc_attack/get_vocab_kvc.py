from network_patch import qwen2_5_7b_instruct_network_emb_layer_1 # Add this line on the top of script.
import vllm_mindspore
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import random


def main(args):
    # Sample prompts.
    # prompts = [
    #     "I am",
    #     "Today is",
    #     "What is"
    # ]

    # traverse all token ids in vocab to obtain all kv cache
    token_ids = [i for i in range(32)]
    token_ids_2 = [i for i in range(150000, 151643)]

    tokens_prompt : TokensPrompt = [
        {"prompt_token_ids": token_ids_2}
    ]
    
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

    # Create an LLM.
    llm = LLM(model=args.model_path, tensor_parallel_size=1)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(tokens_prompt, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def main_loop(args):
    seq_len = 6000
    
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    # Create an LLM.
    llm = LLM(model=args.model_path, tensor_parallel_size=1)
    
    for i in range(25):
        begin, end = seq_len * i, seq_len * (i + 1)
        token_ids = [i for i in range(begin, end)]
        tokens_prompt = {"prompt_token_ids": token_ids}


        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(tokens_prompt, sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    args, _ = parser.parse_known_args()

    # main(args)
    main_loop(args)