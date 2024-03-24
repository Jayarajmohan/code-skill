from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import openai

app = Flask(__name__)

# Load the tokenizer and model
tokenizer_path = 'tokenizer.pkl'
with open(tokenizer_path, 'rb') as file:
    tokenizer = pickle.load(file)

model_path = 'code.h5'
model = keras.models.load_model(model_path)

max_length = 39

def classify_code(code):
    sequence = tokenizer.texts_to_sequences([code])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    return 'beginner' if prediction[0][0] < 0.5 else 'experienced'

def get_suggestion_link(language, code_classification):
    beginner_links = {
        'python': 'https://www.python.org/doc/essays/',#roadmaplinks
        'java': 'https://docs.oracle.com/en/java/',
        'javascript': 'https://developer.mozilla.org/en-US/docs/Web/JavaScript',
    }
    experienced_links = {
        'python': 'https://www.python.org/dev/peps/',
        'java': 'https://docs.oracle.com/javase/specs/',
        'javascript': 'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide',
    }

    if code_classification == 'beginner':
        return beginner_links.get(language, 'https://www.google.com/search?q=beginner+coding+tutorials+{}'.format(language))
    else:
        return experienced_links.get(language, 'https://www.google.com/search?q=advanced+coding+tutorials+{}'.format(language))

def generate_suggestion(code_classification, code_input, language):
    openai.api_key = "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"
    
    if code_classification == 'beginner' or code_classification == 'experienced':
        prompt = f"This {language} code snippet appears to be written by a {code_classification} programmer. find syntax errors, potential inefficiencies,improvements for the below code:\n{code_input}"
    
    chat_completion = openai.ChatCompletion.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-3.5-turbo",
    )
    
    answer = chat_completion.choices[0].message.content
    return answer.strip()

def generate_error_suggestion(code_input, language):
    openai.api_key = "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"
    
    prompt = f"This {language} code snippet appears to be written by a programmer. Get the line by line error suggestion of the below code:\n{code_input}"
    
    chat_completion = openai.ChatCompletion.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-3.5-turbo",
    )
    
    answer = chat_completion.choices[0].message.content
    return answer.strip()

@app.route('/')
def index():
    return render_template('index.html', code_written="")

@app.route('/get_suggestion', methods=['POST'])
def get_suggestion():
    button = request.form['submit']
    code_input = request.form['code_input']
    language = request.form['code_language']
    if(button=='submit'):

        code_classification = classify_code(code_input)
        suggestion_link = get_suggestion_link(language, code_classification)
        suggestion = generate_suggestion(code_classification, code_input,language)
        error_suggestion = ""
    if(button=='check'):
        # return render_template('index.html', code_written="")
        code_classification = classify_code(code_input)
        suggestion_link = ""
        error_suggestion = generate_error_suggestion(code_input,language)
        suggestion = ""
        
    # 

    return render_template('index.html', code_written=code_input, code_classification=code_classification, language=language, suggestion_link=suggestion_link,suggestion=suggestion,error_suggestion=error_suggestion)

if __name__ == '__main__':
    app.run(debug=True)
