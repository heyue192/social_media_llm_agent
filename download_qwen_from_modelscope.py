from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
save_dir = "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("Saved to:", save_dir)
