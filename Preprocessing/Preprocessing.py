# Import necessary packages
import re  
import torch  
import string  
import numpy as np  
import pandas as pd  
from transformers import BertTokenizer, BertModel  
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.model_selection import train_test_split 

# Get BERT embeddings
def GetBertEmbedding(Text, Tokenizer, Model):
    Tokens = Tokenizer(Text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad(): 
        Outputs = Model(**Tokens)  
    return Outputs.last_hidden_state.mean(dim=1).numpy()  

# Reduce each language size using multilingual BERT embeddings and cosine similarity
def ReduceWithBert(DataFrame, MaxSamples=500000):
    Tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    Model = BertModel.from_pretrained('bert-base-multilingual-cased')
    ReducedDataFrame = pd.DataFrame()  
    for Language in DataFrame['Language'].unique():  
        LanguageData = DataFrame[DataFrame['Language'] == Language]
        if len(LanguageData) > MaxSamples: 
            Embeddings = np.vstack(LanguageData['text'].apply(lambda x: GetBertEmbedding(x, Tokenizer, Model)))
            UniqueIndices = set() 
            for Index, Embedding in enumerate(Embeddings):
                if len(UniqueIndices) >= MaxSamples:  
                    break
                Distances = cosine_similarity(Embedding.reshape(1, -1), Embeddings)
                SimilarIndices = np.where(Distances > 0.90)[1]  
                UniqueIndices.add(SimilarIndices[0]) 
            LanguageData = LanguageData.iloc[list(UniqueIndices)] 
        ReducedDataFrame = pd.concat([ReducedDataFrame, LanguageData])  
    return ReducedDataFrame

# Clean text by removing HTML tags, punctuation, and extra spaces
def CleanText(Text):
    Text = re.sub(r'<[^>]+>', '', Text)  
    Text = re.sub(f"[{re.escape(string.punctuation)}]", "", Text) 
    Text = re.sub(r'\s+', ' ', Text).strip() 
    return Text

# Split the dataset into training, validation, and test sets using stratified
def SplitDataset(DataFrame):
    XTrain, XTest, YTrain, YTest = train_test_split(
        DataFrame['text'], DataFrame['Language'], stratify=DataFrame['Language'], test_size=0.15, random_state=42
    )
    XTrain, XVal, YTrain, YVal = train_test_split(
        XTrain, YTrain, stratify=YTrain, test_size=0.15, random_state=42
    )
    TrainDataFrame = pd.DataFrame({'text': XTrain, 'Language': YTrain})  
    ValDataFrame = pd.DataFrame({'text': XVal, 'Language': YVal})  
    TestDataFrame = pd.DataFrame({'text': XTest, 'Language': YTest})  
    return TrainDataFrame, ValDataFrame, TestDataFrame

def main():
    # Loads the dataset
    DataFrame = pd.read_csv("sentences.csv", sep="\t", header=None, names=["Id", "Language", "text"], encoding="utf-8")
    # Remove missing and duplicate entries
    DataFrame = DataFrame.dropna().drop_duplicates().drop(columns=['Id']) 
    DataFrame = DataFrame[DataFrame["Language"] != '\\N']
    # Keep only languages with at least 200 samples
    LanguageCounts = DataFrame["Language"].value_counts()  
    print("Initial Number Of Languages:", len(LanguageCounts))
    LanguagesToKeep = LanguageCounts[LanguageCounts >= 200].index  
    FilteredDataFrame = DataFrame[DataFrame["Language"].isin(LanguagesToKeep)].reset_index(drop=True)
    # Reduce the dataset using BERT embeddings
    BalancedDataFrame = ReduceWithBert(FilteredDataFrame) 
    # Clean the text
    BalancedDataFrame['text'] = BalancedDataFrame['text'].apply(CleanText)  
    # Split into train,val,test sets
    TrainDataFrame, ValDataFrame, TestDataFrame = SplitDataset(BalancedDataFrame)  

    # Save the datasets to CSV files
    TrainDataFrame.to_csv("TrainSentences.csv", index=False, encoding="utf-8")
    ValDataFrame.to_csv("ValSentences.csv", index=False, encoding="utf-8")
    TestDataFrame.to_csv("TestSentences.csv", index=False, encoding="utf-8")


main()
