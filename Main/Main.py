import fasttext
import re
import string
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import json

# Load the FastText model
FasttextModel = fasttext.load_model("BestModelFastText.bin")

# Load the XLM-RoBERTa model and tokenizer
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
XlmRobertaModel = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=197)
XlmRobertaModel.load_state_dict(torch.load('xlm_roberta_best_model.pth'))
XlmRobertaModel.to(Device)
XlmRobertaModel.eval()
XlmRobertaTokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Load the label mapping for XLM-RoBERTa
with open('label_mapping.json', 'r') as f:
    LabelMapping = json.load(f)
Id2label = LabelMapping['id2label']

# Read language dictionary from file
def LoadLanguageDictionary(FilePath):
    with open(FilePath, 'r', encoding='utf-8') as f:
        LanguageDict = json.load(f)  
    return LanguageDict

# Load the languages dictionary
LanguagesDictComplete = LoadLanguageDictionary('languages_dictionary_complete.txt')

# Preprocess text 
def PreprocessText(Text):
    Text = re.sub(r'<[^>]+>', '', Text)  
    Text = re.sub(f"[{re.escape(string.punctuation)}]", "", Text)  
    Text = re.sub(r'\s+', ' ', Text).strip()  
    return Text

# Predict language using FastText model
def PredictLanguageFasttext(Text):
    CleanedText = PreprocessText(Text)
    Label, _ = FasttextModel.predict(CleanedText)
    LanguageCode = Label[0].replace("__label__", "")
    return LanguageCode

# Predict language using XLM-RoBERTa model
def PredictLanguageXlmroberta(Text):
    Text = PreprocessText(Text)
    Inputs = XlmRobertaTokenizer(Text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(Device)
    with torch.no_grad():
        Outputs = XlmRobertaModel(**Inputs)
        Logits = Outputs.logits
        PredictedClass = torch.argmax(Logits, dim=-1)
    return PredictedClass.item()

# Get the language name from language code using the dictionary
def GetLanguageNameFromCode(LanguageCode):
    return LanguagesDictComplete.get(LanguageCode, "Unknown")

if __name__ == "__main__":
    #handle user input and model selection
    print("Choose the model for language detection:")
    print("1. FastText")
    print("2. XLM-RoBERTa")
    
    ModelChoice = input("Enter the number of the model you want to use (1 or 2): ")
    UserInput = input("Please enter a sentence to predict its language: ")
    
    if ModelChoice == '1':
        LanguageCode = PredictLanguageFasttext(UserInput)
        LanguageName = GetLanguageNameFromCode(LanguageCode)
        print(f"The language is: {LanguageName}")
    
    elif ModelChoice == '2':
        PredictedLanguageId = PredictLanguageXlmroberta(UserInput)
        LanguageCode = Id2label.get(str(PredictedLanguageId), "unknown")
        LanguageName = GetLanguageNameFromCode(LanguageCode)
        print(f"The language is: {LanguageName}")
    
    else:
        print("Invalid choice. Please enter 1 or 2.")
