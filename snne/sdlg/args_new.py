import os
import yaml


class Args:
    def __init__(self):

        # MODIFIED: Updated model and dataset lists to match your Semantic Uncertainty framework.
        # The 'deberta_model' is kept from the original SDLG but is not used in your setup.
        self.llm_model = ['llama', 'falcon', 'mistral', 'phi', 'gemma', 'qwen', 'deepseek']
        self.dataset = ['squad', 'trivia_qa', 'nq', 'svamp', 'bioasq']
        self.deberta_model = ["deberta-large-mnli"] # Kept from original SDLG for reference.

        # MODIFIED: A generic run_id. You should set this dynamically based on the model and dataset.
        self.run_id = 'llama2-7b_squad'

        # (1.1) General settings
        # MODIFIED: Renamed from num_total_generations to num_generations to match your script's naming.
        # This corresponds to the number of HIGH-temperature samples. Your script will generate
        # (1 + num_generations) total samples per question.
        self.num_generations = 10
        self.seed_value = 42
        self.max_length_of_generated_sequence = 256
        self.store_logits = True # Kept from original, useful for analysis.

        # (1.2) Low-Temperature Generation (for Accuracy Calculation)
        # MODIFIED: This section now configures the single, low-temperature generation used to
        # calculate the accuracy, as described in your framework.
        self.do_sample_most_likely = True           # Use sampling for generation.
        self.temperature_most_likely = 0.1          # Set to low temperature as in your script.
        self.top_p_most_likely = 1.0                # Corresponds to min_p = 0.0 in your script.
        self.num_beams_most_likely = 1              # Disable beam search when sampling.
        self.num_return_sequences_most_likely = 1   # We only need one most-likely answer.


        # (2.1) High-Temperature Generation (for Uncertainty Calculation)
        # MODIFIED: This section is for the high-temperature, random sampling to generate
        # multiple diverse answers for uncertainty estimation.
        # It replaces the original 'MS (Semantic Entropy)' section.
        self.do_sample_high_temp = True
        # MODIFIED: Corresponds to `args.temperature` in your script for high-temp samples.
        self.temperature_high_temp = 1.0
        # MODIFIED: Corresponds to `args.min_p` in your script. A top_p of 0.95 is a common default.
        self.top_p_high_temp = 0.95
        self.num_beams_high_temp = 1              # Disable beam search for random sampling.


        # (2.2) SDLG Specific Parameters
        # MODIFIED: These parameters are specific to the SDLG method itself.
        # You can keep them if you plan to run SDLG, otherwise they can be ignored.
        self.num_beams_sdlg = 5
        self.token_prob_threshold = 0.001
        self.alphas = (1/3, 1/3, 1/3)  # weighting of attribution, substitution, and importance scores


    def args_to_yaml(self, base_path):
        """Saves the arguments to a YAML file."""
        os.makedirs(base_path, exist_ok=True)
        serializable_attrs = {k: v for k, v in self.__dict__.items()}
        with open(os.path.join(base_path, f'config.yaml'), 'w') as file:
            yaml.dump(serializable_attrs, file, sort_keys=False)