from flask import Flask, render_template, request, jsonify
import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer
import re

app = Flask(__name__)

class HinglishTranslator:
    def __init__(self):
        self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.tokenizer = MBartTokenizer.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.lang_codes = {
            'en': 'en_XX',
            'hi': 'hi_IN'
        }

    def detect_script(self, text):
        devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        return bool(devanagari_pattern.search(text))

    def translate(self, text, target_lang='en'):
        try:
            has_devanagari = self.detect_script(text)
            source_lang = 'hi' if has_devanagari else 'en'
            self.tokenizer.src_lang = self.lang_codes[source_lang]
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang_codes[target_lang]],
                max_length=128,
                num_beams=5,
                length_penalty=1.0
            )
            translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return translation
        except Exception as e:
            return f"Translation error: {str(e)}"

    def translate_hinglish(self, text):
        try:
            translation = self.translate(text, target_lang='en')
            if not translation or translation.startswith("Translation error"):
                words = text.split()
                translated_words = []
                for word in words:
                    if not word.isalnum():
                        translated_words.append(word)
                        continue
                    if self.detect_script(word):
                        word_translation = self.translate(word, target_lang='en')
                        translated_words.append(word_translation)
                    else:
                        translated_words.append(word)
                translation = ' '.join(translated_words)
            return translation
        except Exception as e:
            return f"Translation error: {str(e)}"

# Initialize translator
translator = HinglishTranslator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    text = request.json['text']
    if not text.strip():
        return jsonify({'error': 'Please enter text to translate'})
    translation = translator.translate_hinglish(text)
    return jsonify({'translation': translation})

if __name__ == '__main__':
    app.run(debug=True)