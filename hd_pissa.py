import torch
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import Process
from typing import Dict, Sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from dataclasses import dataclass
import transformers
import os
import math
import copy
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import datetime
import time
import socket

os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def get_hadamard(rank):
    import math
    assert rank > 0 and math.log2(rank).is_integer(), 'rank should be a power of 2'
    def hadamard_matrix(n):
        if n == 1:
            return torch.tensor([[1]], dtype=torch.float32)
        H = hadamard_matrix(n // 2)
        return torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
 
    H = hadamard_matrix(rank)
    return H / (rank ** 0.5)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_custom_model(model, tokenizer, model_path):
    """Save model by temporarily replacing CustomLinearLayer with nn.Linear"""
    os.makedirs(model_path, exist_ok=True)
    
    # Store original layers for restoration
    modules_to_restore = {}
    for name, module in model.named_modules():
        if isinstance(module, CustomLinearLayer):
            modules_to_restore[name] = module
    
    # Replace CustomLinearLayer with nn.Linear
    for name, custom_layer in modules_to_restore.items():
        weight = custom_layer.merge_weights()
        bias = custom_layer.bias
        new_layer = nn.Linear(custom_layer.in_features, custom_layer.out_features)
        new_layer.weight.data = weight
        if bias is not None:
            new_layer.bias = bias
        else:
            new_layer.bias = None
        parent_module = get_parent_module(model, name)
        setattr(parent_module, name.split('.')[-1], new_layer)
    
    # Save model and tokenizer using save_pretrained method
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.save_pretrained(model_path)
    else:
        model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Restore original CustomLinearLayer layers
    for name, custom_layer in modules_to_restore.items():
        parent_module = get_parent_module(model, name)
        setattr(parent_module, name.split('.')[-1], custom_layer)

def get_parent_module(model, module_name):
    module_name_parts = module_name.split('.')
    parent = model
    for part in module_name_parts[:-1]:
        parent = getattr(parent, part)
    return parent

def load_custom_model(model_path):
    """Load model from specified path"""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model



class CustomLinearLayer(nn.Module):
    def __init__(self, original_linear, name, device_id, world_size, ranks_per_gpu=None, alpha=0.0, dropout=0.0):
        super(CustomLinearLayer, self).__init__()
        self.name = name
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = alpha // ranks_per_gpu

        # Get original weights and convert to float32 for SVD
        W = original_linear.weight.data.clone().detach().to(torch.float32)
        
        # Perform SVD decomposition
        U, S, V = torch.svd(W)

        total_rank = S.size(0)
        if ranks_per_gpu is None:
            ranks_per_gpu = total_rank // world_size
            
        # HD-PiSSA method: use hadamard matrix
        start_rank = device_id * ranks_per_gpu
        end_rank = (device_id + 1) * ranks_per_gpu 

        S_sub = S[start_rank:end_rank]
        U_sub = U[:, start_rank:end_rank]
        V_sub = V[:, start_rank:end_rank]
        S_sub_sqrt = torch.diag(torch.sqrt(S_sub))
        
        A = torch.mm(S_sub_sqrt, V_sub.t())
        B = torch.mm(U_sub, S_sub_sqrt)

        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.register_buffer('W_res', W.to(original_linear.weight.dtype))

        if original_linear.bias is not None:
            self.bias = original_linear.bias
        else:
            self.bias = None

    def forward(self, x):
        x_activation = x.to(torch.float32)  
        W_res = self.W_res
        output = nn.functional.linear(x, W_res, bias=self.bias) + nn.functional.linear(x_activation, self.dropout(torch.mm(self.B, self.A)*1e-16*self.alpha)).to(self.W_res.dtype)
        return output

    def merge_weights(self):
        merged_weight = (self.W_res).clone().detach()
        return merged_weight
    
    def __repr__(self):
        return (f"CustomLinearLayer(name={self.name}, "
                f"in_features={self.in_features}, out_features={self.out_features})")

def replace_with_custom_layer(model, target_modules, rank, world_size, ranks_per_gpu=None, alpha=0.0, dropout=0.0):
    for name, module in model.named_modules():
        for target_name in target_modules:
            if target_name in name and isinstance(module, nn.Linear):
                parent_module = get_parent_module(model, name)
                setattr(parent_module, name.split('.')[-1], CustomLinearLayer(module, name, rank, world_size, ranks_per_gpu, alpha, dropout))
                break

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True,)for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = -100

    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: object

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def main(rank, max_length, world_size, model_path, output_path, target_modules, ranks_per_gpu=16, batch_size=16, accumulation_steps=1, data_path="", data_split="", dataset_field=[], num_epochs=1, bf16=False, lr=2e-5, dropout=0.0, warmup_steps=0, warmup_ratio=0.0, schedule="cosine", alpha=0.0):

    start_time = time.time()
    print("start time:",datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path,
        model_max_length=max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if bf16==True:
        print("load in bfloat16")
        dtype = torch.bfloat16
    else:
        print("load in float32")
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    model = model.cuda(rank)

    # Load dataset
    raw_train_datasets = load_dataset(data_path,split=data_split)
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": dataset_field[0], "response": dataset_field[1]}
    )

    def filter_invalid_labels(example):
        # Keep only samples where labels are not all -100
        return not all(label == -100 for label in example['labels'])

    tokenized_datasets = train_dataset
    tokenized_datasets = tokenized_datasets.filter(filter_invalid_labels)
    tokenized_datasets = tokenized_datasets.shuffle(seed=42)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Setup DataLoader
    accumulation_steps = accumulation_steps//world_size
    
    sampler = DistributedSampler(tokenized_datasets, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        tokenized_datasets,
        drop_last=True,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=data_collator,
    )
    
    # Replace target modules with HD-PiSSA
    for param in model.parameters():
        param.requires_grad = False
    replace_with_custom_layer(model, target_modules, rank, world_size, ranks_per_gpu=ranks_per_gpu, alpha=alpha, dropout=dropout)
    
    for name, parameters in model.named_parameters():
        print(f"{name}, :, {parameters.size()},{parameters.requires_grad}")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params}")
    
    # Store Adam states in CustomLinearLayer
    for name, layer in model.named_modules():
        if isinstance(layer, CustomLinearLayer):
            layer.m_A = torch.zeros_like(layer.A.data).cuda(rank)
            layer.v_A = torch.zeros_like(layer.A.data).cuda(rank)
            layer.m_B = torch.zeros_like(layer.B.data).cuda(rank)
            layer.v_B = torch.zeros_like(layer.B.data).cuda(rank)

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    t = 0  # Global step count

    # Define learning rate
    initial_lr = lr

    total_steps = num_epochs * len(dataloader) // accumulation_steps
    if warmup_steps == 0 and warmup_ratio > 0:
        warmup_steps = int(warmup_ratio * total_steps)
    current_step = 1
    loss_list = []

    if rank == 0:
        print(f"Start distributed training for {num_epochs} epochs.")

    last_time = time.time() 

    for epoch in range(num_epochs):
        accumulated_loss = 0
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].cuda(rank, non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(rank, non_blocking=True)
            labels = batch['labels'].cuda(rank, non_blocking=True)
                
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss/float(accumulation_steps)

            avg_loss = loss
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss / world_size
            
            accumulated_loss += avg_loss.item()
            loss.backward()
                    
            if (i + 1) % accumulation_steps == 0:
                loss_list.append(accumulated_loss)

                if t < warmup_steps:
                    lr = initial_lr * t / warmup_steps
                else:
                    if schedule == "cosine":
                        lr = 0.5 * initial_lr * (1 + math.cos(math.pi * (t - warmup_steps) / (total_steps - warmup_steps)))
                    else:
                        lr = initial_lr* (1- (t-warmup_steps)/(total_steps-warmup_steps))

                if rank==0:
                    os.makedirs(output_path, exist_ok=True)
                    with open(f'{output_path}/loss.txt','a') as file:
                        file.write(f'Step:{current_step} Loss:{accumulated_loss}\n')
                t += 1
                current_step += 1
                with torch.no_grad():
                    for name, layer in model.named_modules():
                        if isinstance(layer, CustomLinearLayer):
                            # Get gradients
                            grad_A = layer.A.grad.data*1e16
                            grad_B = layer.B.grad.data*1e16
                                
                            # Adam momentum update
                            layer.m_A = beta1 * layer.m_A + (1 - beta1) * grad_A
                            layer.v_A = beta2 * layer.v_A + (1 - beta2) * (grad_A ** 2)

                            layer.m_B = beta1 * layer.m_B + (1 - beta1) * grad_B
                            layer.v_B = beta2 * layer.v_B + (1 - beta2) * (grad_B ** 2)

                            m_A_hat = layer.m_A / (1 - beta1 ** t)
                            v_A_hat = layer.v_A / (1 - beta2 ** t)
                            m_B_hat = layer.m_B / (1 - beta1 ** t)
                            v_B_hat = layer.v_B / (1 - beta2 ** t)

                            # Calculate delta_A and delta_B
                            delta_A = lr * m_A_hat / (torch.sqrt(v_A_hat) + epsilon)
                            delta_B = lr * m_B_hat / (torch.sqrt(v_B_hat) + epsilon)

                            A_prev = layer.A.data
                            B_prev = layer.B.data

                            # HD-PiSSA weight update
                            delta_A_list = [torch.zeros_like(delta_A) for _ in range(world_size)]
                            delta_B_list = [torch.zeros_like(delta_B) for _ in range(world_size)]
                            dist.all_gather(delta_A_list, delta_A)
                            dist.all_gather(delta_B_list, delta_B)

                            A_prev_list = [torch.zeros_like(A_prev) for _ in range(world_size)]
                            B_prev_list = [torch.zeros_like(B_prev) for _ in range(world_size)]
                            dist.all_gather(A_prev_list, A_prev)
                            dist.all_gather(B_prev_list, B_prev)

                            delta_W_res = torch.zeros_like(layer.W_res)
                                    
                            for i in range(world_size):    
                                delta_W_res -= (delta_B_list[i] @ A_prev_list[i] + B_prev_list[i] @ delta_A_list[i] - delta_B_list[i] @ delta_A_list[i])

                            layer.W_res.data += delta_W_res.to(layer.W_res.dtype)

                            # Zero gradients
                            layer.A.grad = None
                            layer.B.grad = None

                accumulated_loss = 0

        if current_step % 10 == 0 and rank == 0:
            current_time = time.time()
            elapsed_time = current_time - last_time
            last_time = current_time  # Reset timer

            print(f"Step {current_step}/{total_steps} completed, remaining: {total_steps - current_step} steps.")
            print(f"Time for last 10 steps: {elapsed_time:.2f} seconds.")
            print(f"GPU {rank} processing step {current_step}, Loss: {loss_list[-1]}")
        if current_step % 500 == 0 and rank == 0:
            model_path = os.path.join(output_path, f"saved_model_step_{current_step}")
            ensure_dir(os.path.dirname(model_path))
            save_custom_model(model, tokenizer, model_path)
            print(f"Model saved at step {current_step}")

        if rank == 0:
            print(f"Epoch {epoch + 1} completed.")
            model_path = os.path.join(output_path, f"saved_model_step_{current_step}")
            ensure_dir(os.path.dirname(model_path))
            save_custom_model(model, tokenizer, model_path)
            print(f"Model saved at step {current_step}")

    # Save loss list
    if rank == 0:
        loss_list_path = os.path.join(output_path, "loss_list.pkl")
        with open(loss_list_path, 'wb') as f:
            pickle.dump(loss_list, f)

    dist.destroy_process_group()
    end_time = time.time()
    print("end time:",datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"Time elapsed: {end_time - start_time:.2f} seconds.")

import socket
def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        _, port = s.getsockname()
    return str(port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HD-PiSSA Training Script')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model Path')
    parser.add_argument('--output_path', type=str, default='./output', help='Output Path')
    parser.add_argument('--data_path',type=str,default="meta-math/MetaMathQA",help="Data path")
    parser.add_argument('--data_split',type=str,default="train",help="Data split")
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--dataset_field', type=str, default="", help='Dataset field names separated by space')
    parser.add_argument('--target_modules', type=str, default="q_proj o_proj k_proj v_proj gate_proj up_proj down_proj", help='Target modules to replace')
    parser.add_argument('--ranks_per_gpu', type=int, default=16, help='Ranks per GPU')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--bf16',type=bool,default=False,help='Use bfloat16 precision')
    parser.add_argument('--max_length',type=int,default=512,help='Maximum sequence length')
    parser.add_argument('--lr',type=float,default=2e-5,help="Learning rate")
    parser.add_argument('--dropout',type=float,default=0.0,help="Dropout rate")
    parser.add_argument('--warmup_steps',type=int,default=0,help="Warmup steps")
    parser.add_argument('--warmup_ratio',type=float,default=0,help="Warmup ratio")
    parser.add_argument('--schedule',type=str,default="cosine",help="Learning rate schedule")
    parser.add_argument('--alpha',type=float,default=0,help="Alpha parameter for HD-PiSSA")
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = find_free_port()
    dataset_field = args.dataset_field.split()
    target_modules = args.target_modules.split()
    print("Dataset fields:", dataset_field)
    print("Target modules:", target_modules)
    processes = []
    for rank in range(args.world_size):
        p = Process(target=main, args=(
            rank, args.max_length, args.world_size, args.model_path, args.output_path, target_modules,
            args.ranks_per_gpu, args.batch_size, args.accumulation_steps, 
            args.data_path, args.data_split, dataset_field, args.num_epochs, args.bf16, args.lr, args.dropout,
            args.warmup_steps, args.warmup_ratio, args.schedule, args.alpha
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()