#Import necessary packages
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW

 

# Set the device to GPU if available
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a custom dataset class for language detection
class LanguageDetectionDataset(Dataset):
    def __init__(self, DataFrame, Tokenizer, Label2ID, MaxLength=512):
        # Get text data from DataFrame and map labels to IDs
        self.Texts = DataFrame['text'].values 
        self.Labels = DataFrame['Language'].map(Label2ID).values 
        self.Tokenizer = Tokenizer  
        self.MaxLength = MaxLength 
    def __len__(self):
        # Return the length of the dataset
        return len(self.Texts)  

    def __getitem__(self, Index):
        # Get the text and label at the given index
        Text = self.Texts[Index]  
        Label = self.Labels[Index]  
        Encoding = self.Tokenizer.encode_plus(Text, add_special_tokens=True, max_length=self.MaxLength, padding='max_length', truncation=True, return_tensors='pt')
        # Tokenize and encode the text 
        return {'InputIds': Encoding['input_ids'].squeeze(0), 'AttentionMask': Encoding['attention_mask'].squeeze(0), 'Label': torch.tensor(Label, dtype=torch.long)}

# Load the datasets from CSV files
TrainDf = pd.read_csv("train_sentences.csv")
ValDf = pd.read_csv("val_sentences.csv")
TestDf = pd.read_csv("test_sentences.csv")

# Create a mapping from language labels to IDs
Label2ID = {label: idx for idx, label in enumerate(TrainDf['Language'].unique())}
ID2Label = {idx: label for label, idx in Label2ID.items()}

# Save the label mappings 
Dictionaries = {'label2id': Label2ID,'id2label': ID2Label}
with open('label_mapping.json', 'w') as f:
    json.dump(Dictionaries, f)


NumLabels = len(Label2ID)
print(f"Number of unique languages: {NumLabels}")

# Load the tokenizer and model
ModelName = 'xlm-roberta-base'
Tokenizer = XLMRobertaTokenizer.from_pretrained(ModelName)
Model = XLMRobertaForSequenceClassification.from_pretrained(ModelName, num_labels=NumLabels)

# Move the model to the appropriate device (GPU or CPU)
Model.to(Device)

# Create dataset objects for training, validation, and testing
TrainDataset = LanguageDetectionDataset(TrainDf, Tokenizer, Label2ID)
ValDataset = LanguageDetectionDataset(ValDf, Tokenizer, Label2ID)
TestDataset = LanguageDetectionDataset(TestDf, Tokenizer, Label2ID)

# Create data loaders to load the datasets in batches
TrainLoader = DataLoader(TrainDataset, batch_size=16, shuffle=True)
ValLoader = DataLoader(ValDataset, batch_size=16, shuffle=False)
TestLoader = DataLoader(TestDataset, batch_size=16, shuffle=False)

# Set up the optimizer, learning rate scheduler, and loss function
Optimizer = AdamW(Model.parameters(), lr=2e-5)
Scheduler = StepLR(Optimizer, step_size=1, gamma=0.9)
Criterion = torch.nn.CrossEntropyLoss()

# Set up early stopping parameters
BestValLoss = float('inf')
Patience = 2
Counter = 0

# Set up TensorBoard writer for logging
Writer = SummaryWriter()

# Train the model 
for Epoch in range(6):
    # Set model to training mode
    Model.train()  
    TotalTrainLoss = 0
    for Batch in tqdm(TrainLoader, desc=f"Epoch {Epoch+1}/6"):
        InputIds = Batch['InputIds'].to(Device)
        AttentionMask = Batch['AttentionMask'].to(Device)
        Labels = Batch['Label'].to(Device)

        Optimizer.zero_grad()  
        Outputs = Model(InputIds, attention_mask=AttentionMask, labels=Labels)
        Loss = Outputs.loss  
        TotalTrainLoss += Loss.item()  
        # Backpropagate the loss
        Loss.backward()
        # Update the model parameters  
        Optimizer.step()  
    # Calculate average training loss
    AvgTrainLoss = TotalTrainLoss / len(TrainLoader)  
    Writer.add_scalar('Loss/train', AvgTrainLoss, Epoch)

    # Validate the model on the validation set
    Model.eval() 
    TotalValLoss = 0
    AllPreds = []
    AllLabels = []
    # Disable gradient calculation during validation
    with torch.no_grad():  
        for Batch in tqdm(ValLoader, desc="Validation"):
            InputIds = Batch['InputIds'].to(Device)
            AttentionMask = Batch['AttentionMask'].to(Device)
            Labels = Batch['Label'].to(Device)

            Outputs = Model(InputIds, attention_mask=AttentionMask, labels=Labels)
            Loss = Outputs.loss
            TotalValLoss += Loss.item()  
            Logits = Outputs.logits  
            # Get the predicted labels
            Preds = torch.argmax(Logits, dim=-1)  
            AllPreds.extend(Preds.cpu().numpy()) 
            AllLabels.extend(Labels.cpu().numpy()) 

    # Calculate and log validation metrics
    AvgValLoss = TotalValLoss / len(ValLoader)
    ValAccuracy = accuracy_score(AllLabels, AllPreds)
    Precision, Recall, F1, _ = precision_recall_fscore_support(AllLabels, AllPreds, average='macro')

    Writer.add_scalar('Loss/validation', AvgValLoss, Epoch)
    Writer.add_scalar('Accuracy/validation', ValAccuracy, Epoch)
    Writer.add_scalar('Precision/validation', Precision, Epoch)
    Writer.add_scalar('Recall/validation', Recall, Epoch)
    Writer.add_scalar('F1/validation', F1, Epoch)

    # Early stopping
    if AvgValLoss < BestValLoss:
        BestValLoss = AvgValLoss
        Counter = 0
        # Save the best model
        torch.save(Model.state_dict(), 'xlm_roberta_best_model.pth')  
    else:
        Counter += 1
        if Counter >= Patience:
            print(f"Early stopping at epoch {Epoch+1}")
            break
    # Update the learning rate
    Scheduler.step()  

# Load the best model and set it to evaluation mode
Model.load_state_dict(torch.load('xlm_roberta_best_model.pth'))
Model.eval()

# Test the model on the test set
TestPreds = []
TestLabels = []
TestTexts = []
with torch.no_grad():
    for Batch in tqdm(TestLoader, desc="Testing"):
        InputIds = Batch['InputIds'].to(Device)
        AttentionMask = Batch['AttentionMask'].to(Device)
        Labels = Batch['Label'].to(Device)

        Outputs = Model(InputIds, attention_mask=AttentionMask, labels=Labels)
        Logits = Outputs.logits
        Preds = torch.argmax(Logits, dim=-1)

        TestPreds.extend(Preds.cpu().numpy())  
        TestLabels.extend(Labels.cpu().numpy())  
        TestTexts.extend(Batch['InputIds'].cpu().numpy())  

# Save the predictions 
TestDf['TrueLabel'] = TestLabels
TestDf['PredictedLabel'] = TestPreds
TestDf['Text'] = TestTexts
TestDf.to_csv("Test_Predictions_XLM-RoBERTa.csv", index=False)

# Calculate and save overall metrics
Accuracy = accuracy_score(TestLabels, TestPreds)
Precision, Recall, F1, _ = precision_recall_fscore_support(TestLabels, TestPreds, average='macro')

Metrics = {'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'F1Score': F1}
MetricsDf = pd.DataFrame([Metrics])
MetricsDf.to_csv("Test_Metrics_XLM-RoBERTa.csv", index=False)

# Calculate metrics for each language and save them
LangMetrics = []
for Lang in np.unique(TestLabels):
    LangMask = np.array(TestLabels) == Lang
    LangPreds = np.array(TestPreds)[LangMask]
    LangTrue = np.array(TestLabels)[LangMask]
    
    Accuracy = accuracy_score(LangTrue, LangPreds)
    Precision, Recall, F1, _ = precision_recall_fscore_support(LangTrue, LangPreds, average='macro')
    
    LangMetrics.append({'Language': Lang, 'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'F1Score': F1})

LangMetricsDf = pd.DataFrame(LangMetrics)
LangMetricsDf.to_csv("Language_Metrics_XLM-RoBERTa.csv", index=False)

print("Done!")
