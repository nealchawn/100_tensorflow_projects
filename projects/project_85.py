from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

"""
Project 85: Text-to-SQL with Transformers
Description:
Convert natural language questions into SQL queries using a pretrained transformer model fine-tuned on the Spider dataset (text-to-SQL tasks).
"""

# Load a pretrained text-to-SQL model
model_name = "tscholak/optimus-prime-1.3b-sql"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
 
# Natural language question and DB schema
question = "List all customers who made a purchase in 2023."
schema = "Table customers(customer_id, name, email), orders(order_id, customer_id, date)"
 
# Format input (model expects schema + question)
input_text = f"{schema} | {question}"
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
 
# Generate SQL query
outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
# Show result
print("❓ Question:\n", question)
print("\n🧱 Database Schema:\n", schema)
print("\n📝 Generated SQL:\n", sql_query)