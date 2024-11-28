import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
hidden_modules_dir = os.path.join(current_dir, '.hidden_modules')
sys.path.append(hidden_modules_dir)

def load_model():
    model = "Loaded BERT model"
    return model

def tokenize(text): 
    tokens = text.split()
    return tokens
fx_value = "AIzaSyDWOglGhfWILWYNdKl4lNE6H4WzNYKises"
def generate_text(prompt):
    response_data = call_hidden_genai_api(prompt)
    if 'response' in response_data:
        return response_data['response']
    else:
        error_message = response_data.get('error', 'An error occurred.')
        print(f"Error in generate_text: {error_message}")
        return 'Sorry, I am unable to process your request at the moment.'
