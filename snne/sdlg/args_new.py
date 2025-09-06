import os
import yaml


class Args:
    def __init__(self):

        self.llm_family = ['llama', 'falcon', 'mistral', 'phi', 'gemma', 'qwen', 'deepseek'][0]
        self.dataset = ['coqa', 'trivia_qa', 'truthful_qa'][0]

        # 具体 HF 模型 ID（最关键！）
        # 例如：
        # - "meta-llama/Llama-2-7b-chat-hf"
        # - "meta-llama/Llama-3.1-8B-Instruct"
        # - "mistralai/Mistral-7B-Instruct-v0.3"
        # - "tiiuae/falcon-7b-instruct"
        # - "microsoft/Phi-3-mini-4k-instruct"
        # - "google/gemma-2-9b-it"
        # - "Qwen/Qwen2-7B-Instruct"
        # - "deepseek-ai/deepseek-llm-7b-chat"
        self.llm_model_id = "meta-llama/Llama-2-7b-chat-hf"

        self.deberta_model = ["deberta-base-mnli", "deberta-large-mnli", "deberta-xlarge-mnli", "deberta-v2-xlarge-mnli", "deberta-v2-xxlarge-mnli"][1]


        self.run_id = 'llama2-7b_squad'

        # (1.1) General settings
        self.num_generations = 10
        self.seed_value = 42
        self.max_length_of_generated_sequence = 256
        self.store_logits = True # Kept from original, useful for analysis.

        # (1.2) Low-Temperature Generation (for Accuracy Calculation)
        self.do_sample_most_likely = True           # Use sampling for generation.
        self.temperature_most_likely = 0.1          # Set to low temperature as in your script.
        self.top_p_most_likely = 1.0                # Corresponds to min_p = 0.0 in your script.
        self.num_beams_most_likely = 1              # Disable beam search when sampling.
        self.num_return_sequences_most_likely = 1   # We only need one most-likely answer.


        # (2.1) High-Temperature Generation (for Uncertainty Calculation)
        self.do_sample_high_temp = True
        self.temperature_high_temp = 1.0
        self.top_p_high_temp = 0.95
        self.num_beams_high_temp = 1              # Disable beam search for random sampling.


        # (2.2) SDLG Specific Parameters
        self.num_beams_sdlg = 5
        self.token_prob_threshold = 0.001
        self.alphas = (1/3, 1/3, 1/3)  # weighting of attribution, substitution, and importance scores


    def args_to_yaml(self, base_path):
        """Saves the arguments to a YAML file."""
        os.makedirs(base_path, exist_ok=True)
        serializable_attrs = {k: v for k, v in self.__dict__.items()}
        with open(os.path.join(base_path, f'config.yaml'), 'w') as file:
            yaml.dump(serializable_attrs, file, sort_keys=False)