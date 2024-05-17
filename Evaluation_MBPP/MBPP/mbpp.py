import time
import string
import multiprocessing
import os
import numpy as np
import json
import re
import torch
import datetime
import subprocess
import torch.distributed as dist
from tqdm import tqdm
from human_eval.evaluation import evaluate_functional_correctness
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from utils.dataset import MBPPDataset
from utils.utils import cleanup_code

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_str, tokenizer):
        StoppingCriteria.__init__(self)
        self.current_context = []
        self.tokenizer = tokenizer
        self.keywords_str = keywords_str
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.current_context.append(input_ids[0][-1].item())
        current_context = self.tokenizer.decode(self.current_context)
        for word in self.keywords_str:
            if word in current_context:
                return True
        return False


class MBPP:
    """
    MBPP evaluation class.
    """
    def __init__(self, data_root, max_seq_len=2048,
                language="python", max_gen_len=200, batch_size=512,
                log_dir=None, temperature=0, issft=False, top_p=0.95,
                model_name="", inference_increment=True,
                tokenizer_cfg=None, n_sample=10, k_sample=1):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.batch_size = batch_size
        self.k = k_sample
        self.n_sample = n_sample
        self.language = language
        self.log_dir = log_dir
        self.sft = issft
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = tokenizer_cfg["model_path"].replace("/", "_")
        self.inference_increment = inference_increment
        os.makedirs(self.log_dir, exist_ok=True)
        tokenizer_cls = tokenizer_cfg.pop('cls')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg.pop("model_path"), trust_remote_code=True)       
        except Exception as e:
            print(e)
            assert False

    @torch.no_grad()
    def eval_model(self, gpt, accelerator):
        """
        Evaluate the model.
        """
        assert self.log_dir is not None, "log_dir should not be None when evaluating MBPP"
        print(f"Using model: {gpt}, and logdir {self.log_dir}")
        dataset = MBPPDataset(self.data_root, samplenum=self.n_sample)
        nprompt = len(dataset) // self.n_sample
        dp_rank = accelerator.process_index 
        dp_size = accelerator.num_processes 
        if self.k > 10:
            assert self.n_sample >= 10, "MBPP PASS@100 needs n_sample >= 100"
        gpt.eval()
        prompt_indices_split = np.array_split(range(nprompt), dp_size)
        prompt_indices = prompt_indices_split[dp_rank]
        indices = []
        for x in prompt_indices:
            for j in range(self.n_sample):
                indices.append(x * self.n_sample + j)
        all_num = len(indices)
        print("All_num is: ", all_num)
        processed_num = 0
        log_file = os.path.join(self.log_dir,
                                    f'{self.model_name}_rank{dp_rank}_bs{self.batch_size}_shot_log_{self.language}.json')
        tmpfile = open(log_file, "w")

        totoalnum = 0        
        start_time = time.time()

        for idx in tqdm(range(0, len(indices), self.batch_size)):
            prompt_list = []
            prompt_lens = []
            answers_list = []
            test_list = []
            taskid = []
            tokenized_prompt_lens = []
            for j in indices[idx:idx + self.batch_size]:
                data = dataset[j]
                prompt = dataset.prompt
                prompt1 = data["prompt"]
                tests = "\n".join(data["test"])
                test_list.append(data["test"])
                system = "You are an expert Python programmer, don't explain anything, just write the code."
                user = "This is your task: " + prompt1 + "\nYour code should pass these tests:\n\n" + tests
                if "Gemma" in self.model_name:
                    fprompt = f"<start_of_turn>user\n{system}\n{user}\n<end_of_turn>\n<start_of_turn>model\n"
                elif "Llama-3-8B" in self.model_name:
                    fprompt = f"{system}\n{user}"
                else:
                    fprompt = f"[INST]\n{system}\n{user}\n[/INST]\n"
                prompt_list.append(fprompt)
                answers_list.append(data['code'])
                prompt_lens.append(len(fprompt))
                taskid.append(data["task_id"])
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenized_prompt = self.tokenizer(prompt_list, padding=True, return_tensors="pt")
            inputids = tokenized_prompt["input_ids"].to(gpt.device)[:, -self.max_seq_len:]
            attenion_mask = tokenized_prompt["attention_mask"].to(gpt.device)[:, -self.max_seq_len:]

            if "Gemma" in self.model_name:
                stop_criteria = KeywordsStoppingCriteria(["<end_of_turn>"], self.tokenizer)
            elif "CodeLlama" in self.model_name or "Mistral" in self.model_name:
                stop_criteria = KeywordsStoppingCriteria(["[/INST]", "[END]", "[/PYTHON]", "[```]", "[/SOL]", "[DONE]"], self.tokenizer)
            self.temperature = 0.8
            print("TEMPERATURE IS: ", self.temperature)
            if not "Llama-3-8B" in self.model_name:
                decoded = gpt.generate(
                    input_ids=inputids,
                    attention_mask=attenion_mask,
                    max_new_tokens=self.max_gen_len,
                    top_p=self.top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([stop_criteria]),
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                decoded = gpt.generate(
                    input_ids=inputids,
                    attention_mask=attenion_mask,
                    max_new_tokens=self.max_gen_len,
                    top_p=self.top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            print("finished generating")
            for local_idx, text in enumerate(decoded):
                prediction = decoded[local_idx]
                prediction = self.tokenizer.decode(prediction, skip_special_tokens=True)

                idx_1 = prediction.find("```python")
                idx = prediction.find("[PYTHON]")
                idx0 = prediction.find("[DONE]")
                idx1 = prediction.find("[/PYTHON]")
                idx2 = prediction.find("[SOL]")
                idx2_ = prediction.find("[/SOL]")
                idx3 = prediction.find("```")
                idx3_ = prediction.find("```", idx3 + len("```"))
                idx4 = prediction.find("[/INST]")
                idx6 = prediction.find("[BEGIN]")
                idx5 = prediction.find("[BEGIN]", idx6 + len("[BEGIN]"))
                idx7 = prediction.find("[PY]")
                idx8 = prediction.find("[/PY]")
                print(prediction)
                if idx_1 != -1:
                    prediction = prediction[idx_1 + len("```python"): idx3_]
                elif idx7 != -1:
                    prediction = prediction[idx7 + len("[PY]"): idx8]
                elif idx != -1 and idx1 !=-1:
                    prediction = prediction[idx + len("[PYTHON]"): idx1]
                elif idx != -1 and idx0 !=-1:
                    prediction = prediction[idx + len("[PYTHON]"): idx0]
                elif idx2 != -1:
                    prediction = prediction[idx2 + len("[SOL]"): idx2_]
                elif idx3 != -1:
                    prediction = prediction[idx3 + len("```"): idx3_]
                elif idx5 !=-1 and idx0 != -1:
                    prediction = prediction[idx5 + len("[BEGIN]"): idx0]
                elif idx == -1 and idx2 == -1 and idx3 == -1 and idx5 == -1:
                    prediction = prediction[idx4 + len("[/INST]"):]

                print("PRED: ", prediction)
                res = {"task_id": taskid[local_idx], "generation": prediction}
                tmpfile.write(json.dumps(res) + "\n")
                tmpfile.flush()
                totoalnum += 1

            self.log_score(dp_rank, totoalnum, all_num, start_time, self.batch_size)
        tmpfile.close()
        accelerator.wait_for_everyone()
        self._calculate_final_score(accelerator)

    def log_score(self, dp_rank, processed_num, all_num, start_time, bs):
        """
        Log the score.
        """
        mem = torch.cuda.max_memory_allocated() / (1 << 30)
        avg_time = (time.time() - start_time) / processed_num * bs
        print(
            f'DP RANK:{dp_rank} process_num/all_num:{int(processed_num)}/{all_num} '
            f'avg_time_per_batch:{avg_time:.2f} s '
            f'still_need:{((all_num - processed_num) // bs + 1) * avg_time / 60:.2f} m',
            f'mem:{mem:.3f} GiB bs:{bs}',
            flush=True
        )
        if processed_num == all_num:
            print(f'EVAL DONE! Process time {(time.time() - start_time) / 60:.2f} m', flush=True)

    def _calculate_final_score(self, accelerator):
        """
        Calculate the final score.
        """
        if accelerator.is_local_main_process:
            logfilepath = os.path.join(self.log_dir, f'final_{self.model_name}.jsonl')
            logfile = open(logfilepath, "w")
            for i in range(accelerator.num_processes):
                tmplogfile = os.path.join(self.log_dir, f'{self.model_name}_rank{i}_bs{self.batch_size}_shot_log_{self.language}.json')
                logfile.write(open(tmplogfile).read().strip() + "\n")
                os.remove(tmplogfile)
            logfile.close()
            timeout = 10
            runlang = self.language
            res = evaluate_functional_correctness(input_file=logfilepath, problem_file=os.path.join(self.data_root, f"mbpp_test.jsonl"), tmp_dir=self.log_dir, timeout=timeout, language=runlang)
            print("score is", res['pass@%d' % self.k])
            #os.remove(logfilepath)
        return
            