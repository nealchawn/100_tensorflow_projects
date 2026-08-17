from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

"""
Project 67: Dialogue Generation with DialoGPT
Description:
Use Microsoft’s pretrained DialoGPT model to generate conversational replies in a chatbot-style setting, ideal for casual dialogue generation.
"""

# Load pretrained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
 
# Conversation history (starts with a user prompt)
chat_history_ids = None
user_input = "Hi there! How are you today?"
 
# Encode user input and append to chat history
new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
 
# Generate response
chat_history_ids = model.generate(
    bot_input_ids,
    max_length=1000,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8
)
 
# Decode and print bot response
response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
print("🧑 You:", user_input)
print("🤖 Bot:", response)