import mii
pipe = mii.pipeline("mistralai/Mistral-7B-v0.1")
response = pipe("DeepSpeed is", max_new_tokens=128)
print(response)