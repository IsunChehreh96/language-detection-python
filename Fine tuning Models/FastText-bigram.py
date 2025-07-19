# Import necessary packages
import pandas as pd
import torch
import fasttext
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report

# Check if CUDA is available for using GPU
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {Device}")

# Load necessary data from CSV files
TrainDf = pd.read_csv('train_sentences.csv', encoding="utf-8")
ValDf = pd.read_csv('val_sentences.csv', encoding="utf-8")
TestDf = pd.read_csv('test_sentences.csv', encoding="utf-8")

# Convert data to FastText format 
def SaveFastTextFormat(Df, Filename):
    with open(Filename, 'w') as F:
        for _, Row in Df.iterrows():
            F.write(f"__label__{Row['Language']} {Row['text']}\n")

# Save training and validation data in FastText format
SaveFastTextFormat(TrainDf, 'TrainDataFastText.txt')
SaveFastTextFormat(ValDf, 'ValDataFastText.txt')

# Variables for early stopping
BestValF1 = 0
Patience = 3
NoImprovement = 0

# Train the FastText model 
for Epoch in range(1, 15):
    print(f"Epoch {Epoch} training...")
    # Train model 
    Model = fasttext.train_supervised(input="TrainDataFastText.txt", epoch=Epoch, lr=0.1, wordNgrams=2)
    # Get predictions for validation data
    ValPredictions = [Model.predict(Text)[0][0].replace('__label__', '') for Text in ValDf['text']]
    # Calculate F1 score for the validation data
    ValF1 = f1_score(ValDf['Language'], ValPredictions, average='weighted')
    if ValF1 > BestValF1:
        BestValF1 = ValF1
        Model.save_model(f"BestModelFastText.bin")
        NoImprovement = 0
    else:
        NoImprovement += 1
    
    # Stop early if there is no improvement 
    if NoImprovement >= Patience:
        print(f"Stopping early at epoch {Epoch}.")
        break

# Load the best model after training
Model = fasttext.load_model("BestModelFastText.bin")

# Predict languages for validation data
ValPredictions = [Model.predict(Text)[0][0].replace('__label__', '') for Text in ValDf['text']]
ValDf['PredictedLanguage'] = ValPredictions

# Calculate evaluation metrics for validation
ValAccuracy = accuracy_score(ValDf['Language'], ValDf['PredictedLanguage'])
ValPrecision, ValRecall, ValF1, _ = precision_recall_fscore_support(ValDf['Language'], ValDf['PredictedLanguage'], average='weighted')

# Print validation results
print(f"Validation Accuracy: {ValAccuracy:.4f}")
print(f"Validation Precision: {ValPrecision:.4f}")
print(f"Validation Recall: {ValRecall:.4f}")
print(f"Validation F1-Score: {ValF1:.4f}")

# Predict languages for test data
TestPredictions = [Model.predict(Text)[0][0].replace('__label__', '') for Text in TestDf['text']]
TestDf['PredictedLanguage'] = TestPredictions

# Save test predictions to a CSV file
TestDf[['text', 'Language', 'PredictedLanguage']].to_csv('TestPredictionsFastText.csv', index=False)

# Calculate and save test evaluation metrics
TestAccuracy = accuracy_score(TestDf['Language'], TestDf['PredictedLanguage'])
Precision, Recall, F1, _ = precision_recall_fscore_support(TestDf['Language'], TestDf['PredictedLanguage'], average='weighted')

with open('MetricsOverallFastText.txt', 'w') as F:
    F.write(f"Test Accuracy: {TestAccuracy:.4f}\n")
    F.write(f"Test Precision: {Precision:.4f}\n")
    F.write(f"Test Recall: {Recall:.4f}\n")
    F.write(f"Test F1-Score: {F1:.4f}\n")

# Generate and save a classification report for each language
Report = classification_report(TestDf['Language'], TestDf['PredictedLanguage'], output_dict=True)
pd.DataFrame(Report).transpose().to_csv('LanguageWiseMetricsFastText.csv')

print("Done!")
