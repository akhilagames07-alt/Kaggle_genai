from transformers import pipeline

# Load a small AI model for text generation
generator = pipeline("text-generation", model="gpt2")

# Try your first test prompt
prompt = "explain some motivational quotes."
result = generator(prompt, max_length=80, temperature=0.7)
print(result[0]['generated_text'])
