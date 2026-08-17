import spacy

"""
Project 77: Coreference Resolution using spaCy
Description:
Identify and resolve coreferences in a text—i.e., link pronouns like “he”, “she”, or “it” to the nouns they refer to—using spaCy with a coreference resolution extension.

⚠️ Requires: pip install spacy and pip install coreferee
Then run: python -m coreferee install en
"""

# Load spaCy model with coreferee extension
nlp = spacy.load("en_core_web_sm")
import coreferee
nlp.add_pipe('coreferee')
 
# Input text with pronouns
text = "Angela went to the market. She bought some apples. Then she met her friend."
 
# Run coreference resolution
doc = nlp(text)
 
# Display coreference clusters
print("📄 Original Text:\n", text)
print("\n🔗 Coreference Clusters:")
for chain in doc._.coref_chains:
    mentions = [doc[span.start:span.end].text for span in chain]
    print(" → ".join(mentions))