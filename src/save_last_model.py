
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from step1_bidirectional import patch_to_bidirectional

base_model_path = "./bidir_dnagpt"   # Step1 输出
adapter_ckpt    = "./mntp_dnagpt_lora_fn_s42_r1/checkpoints/checkpoint-68500"  # 改成 best checkpoint
save_dir        = "./mntp_dnagpt_lora_fn_merged_s42_r1"  # 建议先存到新目录，避免覆盖

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=dtype,
    attn_implementation="eager",
)
base = patch_to_bidirectional(base)

model = PeftModel.from_pretrained(base, adapter_ckpt)
merged = model.merge_and_unload()

os.makedirs(save_dir, exist_ok=True)
merged.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Saved merged model to: {save_dir}")

