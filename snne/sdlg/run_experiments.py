import os
# os.environ["HF_HOME"] = ...                     # set accordingly
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # set accordingly

import pickle
import yaml

import numpy as np
import datasets
import evaluate
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset

from args import Args
from utils import seed_everything, get_models_and_tokenizers, compute_correctness, compute_semantic_paris
from utils import generate_text, prepare_generated_text, compute_likelihood, prepare_likelihood
from sdlg import generate_semantically_diverse_output_sequences

from transformers import AutoTokenizer


CUDA_ID_LLM = 0                 # set accordingly
CUDA_ID_DEBERTA = CUDA_ID_LLM   # set accordingly


def encode(examples):
    # 为了拿到 tokenizer，这里假定 tokenizer 在外层已创建（你的代码里已如此）
    if args.dataset == 'coqa':
        # 原始coqa prompt: story + Q: ... A:
        prompts = [s + ' Q: ' + q + ' A:' for s, q in zip(examples['story'], examples['question'])]
    else:
        # trivia_qa / truthful_qa：默认不拼 context（与原仓库 nocontext 设定对齐）
        # 如需加 context，可改为 f"{ctx} Q: {q} A:"
        prompts = ['Q: ' + q + ' A:' for q in examples['question']]

    return tokenizer(prompts, truncation=False, padding=False)


def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset


def get_results(args, base_path, llm_model, tokenizer, device_llm, deberta_model, deberta_tokenizer, device_deberta, dataset):

    squad_metric = evaluate.load("squad")
    rouge = evaluate.load('rouge')
    exact_match_metric = evaluate.load("exact_match")
    # Fix: Use proper BLEU metric or handle sacrebleu correctly
    # try:
    #     bleurt = evaluate.load("bleurt") 
    # except:
    #     # Fallback to sacrebleu if bleurt is not available
    #     bleurt = evaluate.load("sacrebleu")
    #     print("Warning: Using sacrebleu instead of bleurt")
    bleurt = None

    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
    ).squeeze().detach()

    if args.dataset == 'coqa':
        id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))

    dataloader = DataLoader(dataset, batch_size=1)
    total = len(dataloader)
    for b, batch in tqdm(enumerate(dataloader), total=total, desc="Processing samples"):

        prompt = batch['input_ids'][0].to('cpu')

        if args.dataset == 'coqa':
            question = id_to_question_mapping[batch['id'][0]]  
        else:
            question = batch["question"][0]

        results_dict = {'input_ids': batch['input_ids'],
                        'question': question,
                        'correctness_dict': {},
                        'sdlg': {'generations': [],      # list of dicts
                                'likelihoods': []},      # list of dicts
                        'baseline': {'generations': [],  # list of dicts
                                    'likelihoods': []}  # list of dicts
                        }
                
        ### (1) most likely output sequence
        most_likely_generation = generate_text(args=args, 
                                               model=llm_model, 
                                               tokenizer=tokenizer, 
                                               input_ids=batch['input_ids'], 
                                               len_prompt=len(prompt), 
                                               decoding_method='most_likely', 
                                               device=device_llm)
        
        # print("💜", most_likely_generation['generation_ids'])
        # print("💙", most_likely_generation['generation_text'])
        # print("💚", most_likely_generation['cleaned_generation_ids'])
        # print("💛", most_likely_generation['cleaned_generation_text'])

        # compute correctness score
        if args.dataset == 'coqa':
            # Fix: Convert tuple/list format to proper list of strings
            # reference_answers = []
            # if hasattr(batch['answer'], 'text'):
            #     reference_answers.extend(batch['answer']['text'])
            # elif isinstance(batch['answer'], (list, tuple)):
            #     reference_answers.extend([str(ans) for ans in batch['answer']])
            # else:
            #     reference_answers.append(str(batch['answer']))
            reference_answers = batch['answer']['text'] + [x[0] for x in batch['additional_answers']]
            incorrect_answers = []
            
            # Handle additional answers
            # if 'additional_answers' in batch and batch['additional_answers']:
            #     for additional in batch['additional_answers']:
            #         if isinstance(additional, (list, tuple)) and len(additional) > 0:
            #             reference_answers.append(str(additional[0]))
            #         else:
            #             reference_answers.append(str(additional))
            # incorrect_answers = []
            
        elif args.dataset == 'trivia_qa':
            # Fix: Ensure reference_answers is a list of strings
            # if isinstance(batch['answer'], (list, tuple)):
            #     reference_answers = [str(ans) for ans in batch['answer']]
            # else:
            #     reference_answers = [str(batch['answer'])]
            # incorrect_answers = []
            reference_answers = [ans[0] for ans in batch['answer']] # change tuple -> str
            incorrect_answers = []
            
        elif args.dataset == 'truthful_qa':
            reference_answers = batch['answer'] + [x[0] if x[0][-1] == "." else x[0] + "." for x in batch['additional_answers']]
            if "I have no comment." not in reference_answers:
                reference_answers.append("I have no comment.")
            incorrect_answers = [x[0] if x[0][-1] == "." else x[0] + "." for x in batch['incorrect_answers']]
            # Fix: Properly handle reference and incorrect answers
            # reference_answers = []
            # if isinstance(batch['answer'], (list, tuple)):
            #     reference_answers.extend([str(ans) for ans in batch['answer']])
            # else:
            #     reference_answers.append(str(batch['answer']))
            
            # # Handle additional answers
            # if 'additional_answers' in batch and batch['additional_answers']:
            #     for additional in batch['additional_answers']:
            #         if isinstance(additional, (list, tuple)) and len(additional) > 0:
            #             ans_text = str(additional[0])
            #             if not ans_text.endswith("."):
            #                 ans_text += "."
            #             reference_answers.append(ans_text)
            
            # # Add default fallback answer
            # if "I have no comment." not in reference_answers:
            #     reference_answers.append("I have no comment.")
            
            # # Handle incorrect answers
            # incorrect_answers = []
            # if 'incorrect_answers' in batch and batch['incorrect_answers']:
            #     for incorrect in batch['incorrect_answers']:
            #         if isinstance(incorrect, (list, tuple)) and len(incorrect) > 0:
            #             ans_text = str(incorrect[0])
            #             if not ans_text.endswith("."):
            #                 ans_text += "."
            #             incorrect_answers.append(ans_text)

        # Clean and validate the generated text
        # generated_text = most_likely_generation['generation_text'][0]
        # if not generated_text or not isinstance(generated_text, str):
        #     print(f"Warning: Invalid generated text for sample {b}: {generated_text}")
        #     generated_text = ""  # Fallback to empty string
        
        # print("👌most_likely_generated_text:", most_likely_generation['generation_text'][0])
        # print("👌incorrect_answers:", incorrect_answers)
        # print("👌reference_answers:", reference_answers)

        correctness_dict = compute_correctness(args=args, 
                                               reference_answers=reference_answers, 
                                               incorrect_answers=incorrect_answers, 
                                               most_likely_generation_text=most_likely_generation['cleaned_generation_text'][0], # change to cleaned_version 
                                               exact_match_metric=exact_match_metric, 
                                               rouge=rouge, 
                                               bleurt=bleurt)
        
        results_dict['correctness_dict'] = correctness_dict

        # compute likelihood
        most_likely_generation_likelihoods = compute_likelihood(prompt=prompt, 
                                                                generation=most_likely_generation, 
                                                                model=llm_model, 
                                                                device=device_llm, 
                                                                compute_cleaned=args.compute_cleaned, 
                                                                store_logits=args.store_logits)
        # print("😝",most_likely_generation_likelihoods['average_neg_log_likelihood'])

        # (2.1) SDLG
        results_dict['sdlg']['generations'].append(most_likely_generation)
        results_dict['sdlg']['likelihoods'].append(most_likely_generation_likelihoods)
        
        results_dict = generate_semantically_diverse_output_sequences(results_dict=results_dict, 
                                                                      deberta_model=deberta_model, 
                                                                      deberta_tokenizer=deberta_tokenizer, 
                                                                      device_deberta=device_deberta,
                                                                      deberta_embeddings=deberta_embeddings,
                                                                      model=llm_model, 
                                                                      tokenizer=tokenizer, 
                                                                      device_llm=device_llm,
                                                                      input_ids=batch['input_ids'], 
                                                                      prompt=prompt,
                                                                      question=question, 
                                                                      initial_generation=most_likely_generation,
                                                                      initial_likelihood=most_likely_generation_likelihoods,
                                                                      args=args)      

        # (2.2) MS
        assert args.num_total_generations % args.num_return_sequences_baseline == 0
        results_dict['baseline']['generations'].append(most_likely_generation)
        results_dict['baseline']['likelihoods'].append(most_likely_generation_likelihoods)

        for i in range(int(args.num_total_generations / args.num_return_sequences_baseline)):
            baseline_generation = generate_text(args=args, 
                                                model=llm_model, 
                                                tokenizer=tokenizer, 
                                                input_ids=batch['input_ids'], 
                                                len_prompt=len(prompt), 
                                                decoding_method='baseline', 
                                                device=device_llm)

            results_dict['baseline']['generations'].append(baseline_generation)
            results_dict['baseline']['likelihoods'].append(compute_likelihood(prompt=prompt, 
                                                                              generation=baseline_generation, 
                                                                              model=llm_model, 
                                                                              device=device_llm, 
                                                                              compute_cleaned=args.compute_cleaned, 
                                                                              store_logits=args.store_logits))

        with open(os.path.join(base_path, f'results_dict_{b}.pkl'), 'wb') as outfile:
            pickle.dump(results_dict, outfile)


if __name__ == '__main__':

    args = Args()

    base_path = os.path.join('results', args.run_id)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    
    if os.path.exists(os.path.join(base_path, f'config.yaml')):
        with open(os.path.join(base_path, f'config.yaml'), 'r') as file:
            existing_args = yaml.load(file, Loader=yaml.FullLoader)
        changes = False

        for k, v in existing_args.items():
            if k not in args.__dict__:
                print(f"new arg: {k}")
                changes = True
            elif v != args.__dict__[k]:
                print(f"arg {k} changed from {v} to {args.__dict__[k]}")
                changes = True
        if changes:
            exit()
        print("continuing existing run ...")
    else:
        print("starting new run ...")

    # save args
    args.args_to_yaml(base_path)

    print("run_id", args.run_id)

    seed_everything(seed=args.seed_value)

    # prepare model & tokenizer
    device_llm = "mps" if torch.backends.mps.is_built() else f"cuda:{CUDA_ID_LLM}" if torch.cuda.is_available() else "cpu"
    print("device_llm: ", device_llm)
    device_deberta = "mps" if torch.backends.mps.is_built() else f"cuda:{CUDA_ID_DEBERTA}" if torch.cuda.is_available() else "cpu"
    print("device_deberta: ", device_deberta)

    llm_model, tokenizer, deberta_model, deberta_tokenizer = get_models_and_tokenizers(
        model_type_llm=args.llm_model_id,                   
        device_llm=device_llm, 
        model_type_deberta=args.deberta_model, 
        device_deberta=device_deberta,
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(args, 'llm_model_id') and 'llama' in args.llm_model_id.lower():
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # prepare data
    if args.dataset == 'coqa':
        # 先保留本地预处理版本（coqa结构比较复杂，后面需要时再迁到HF）
        dataset = datasets.load_from_disk(os.path.join("datasets", f'coqa_dataset'))
        dataset = encode_and_format_dataset(dataset)


    elif args.dataset == 'trivia_qa':
        # === 直接用 HF: TriviaQA (SQuAD格式) ===
        raw = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        raw = raw.train_test_split(test_size=0.2, seed=args.seed_value)
        dataset = raw['test']  # 和你之前只用validation一致；要跑train改成 raw['train']

        # 切片
        if getattr(args, "max_samples", None):
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            print(f"[Info] limiting TriviaQA to first {len(dataset)} samples")

        def map_trivia(ex):
            ans_list = ex['answers']['text'] if isinstance(ex.get('answers'), dict) else ex.get('answers', [])
            ans_list = [str(a).strip() for a in ans_list if a is not None and str(a).strip() != ""]
            return {
                'id': ex.get('id', ''),
                'question': ex['question'],
                'answer': ans_list
            }
        dataset = dataset.map(map_trivia, remove_columns=[c for c in dataset.column_names if c not in ['id','question','answer']])

        # === few-shot prompt + encode_and_format_dataset ===
        # 构造 few-shot prompt
        train_data = raw['train'].select(range(0, 10))
        few_shot_prompt = 'This is a bot that correctly answers questions. \n'
        for sample in train_data:
            ans_list = sample['answers']['text'] if isinstance(sample['answers'], dict) else sample.get('answers', [])
            answer = ans_list[0] if ans_list else "Unknown"
            few_shot_prompt += f"Q: {sample['question']} A: {answer} "

        tokenizer = AutoTokenizer.from_pretrained(args.llm_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        def encode_and_format_dataset(batch):
            inputs = [few_shot_prompt + "Q: " + q + " A:" for q in batch['question']]
            answers = [a[0] if len(a) > 0 else "" for a in batch['answer']]

            model_inputs = tokenizer(inputs, padding=False, truncation=False)
            model_outputs = tokenizer(answers, padding=False, truncation=False)

            batch['input_ids'] = model_inputs.input_ids
            batch['attention_mask'] = model_inputs.attention_mask
            batch['decoder_input_ids'] = model_outputs.input_ids
            batch['decoder_attention_mask'] = model_outputs.attention_mask
            batch['labels'] = model_outputs.input_ids.copy()

            # 忽略 pad token 的 loss
            batch['labels'] = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch['labels']
            ]
            return batch

        dataset = dataset.map(encode_and_format_dataset,
                            batched=True,
                            batch_size=1)

        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
            output_all_columns=True)


    elif args.dataset == "truthful_qa":
        # === 直接用 HF: TruthfulQA-generation 版本 ===
        raw = datasets.load_dataset("truthful_qa", "generation")['validation']
        # 规范化到仓库现有字段：question, answer(list[str]), additional_answers(list[str]), incorrect_answers(list[str])
        def map_tqa(ex):
            # HF里通常有 'best_answer', 'correct_answers', 'incorrect_answers'
            best = [str(ex['best_answer']).strip()] if ex.get('best_answer') else []
            correct = [str(s).strip() for s in (ex.get('correct_answers', []) or [])]
            # 把 best + correct 合并成 "正确集合"
            ref = [s for s in (best + correct) if s.strip()]
            inc = [str(s).strip() for s in (ex.get('incorrect_answers', []) or []) if str(s).strip()]
            
            return {
                'id': ex.get('id', ''),
                'question': ex['question'],
                'answer': ref,                  # list[str]
                'additional_answers': [],       # 先空表（仓库逻辑里会额外append "I have no comment."）
                'incorrect_answers': inc        # list[str]
            }
        dataset = raw.map(map_tqa, remove_columns=[c for c in raw.column_names if c not in ['id','question','answer','additional_answers','incorrect_answers']])
        dataset = encode_and_format_dataset(dataset)

    else:
        raise ValueError(f"dataset {args.dataset} not implemented")

    print("# dataset:", len(dataset))

    get_results(args=args,
                base_path=base_path, 
                llm_model=llm_model, 
                tokenizer=tokenizer, 
                device_llm=device_llm, 
                deberta_model=deberta_model, 
                deberta_tokenizer=deberta_tokenizer, 
                device_deberta=device_deberta, 
                dataset=dataset)

    compute_semantic_paris(base_path=base_path, 
                           model_type=args.deberta_model, 
                           deberta_tokenizer=deberta_tokenizer, 
                           deberta_model=deberta_model, 
                           num_instances=len(dataset),
                           device=device_deberta)