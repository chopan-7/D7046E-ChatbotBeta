import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
import nltk
import random
import requests
import json

nltk.download('punkt')
nltk.download('stopwords')

class ChatbotDataset(Dataset):
    def __init__(self, csv_file, root_dir="./", transform=None):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000, max_df=0.5, use_idf=True, norm='l2')
        self.vocab_size = 0
        
        # Data and labels
        self.training_data = []
        self.validation_data = []
        self.training_labels = []
        self.validation_labels = []
        self.training_dataset = []
        self.validation_dataset = []

        # Tensors
        self.train_x_tensor = []
        self.train_y_tensor = []
        self.validation_x_tensor = []
        self.validation_y_tensor = []

        # DataLoaders
        self.training_loader = []
        self.validation_loader = []

    def create(self):
        print("Creating dataset...")

        data = pd.read_csv(self.csv_file, delimiter='\t', header=None)
        data.columns = ['Sentence', 'Class']
        data['index'] = data.index                                          # add new column index
        columns = ['index', 'Class', 'Sentence']
        data = self.preprocess_pandas(data, columns)                             # pre-process
        self.training_data, self.validation_data, self.training_labels, self.validation_labels = train_test_split( # split the data into training, validation, and test splits
            data['Sentence'].values.astype('U'),
            data['Class'].values.astype('int32'),
            test_size=0.20,
            random_state=0,
            shuffle=True
        )

        # vectorize data using TFIDF and transform for PyTorch for scalability
        self.training_data = self.word_vectorizer.fit_transform(self.training_data)        # transform texts to sparse matrix
        self.training_data = self.training_data.todense()                             # convert to dense matrix for Pytorch
        self.vocab_size = len(self.word_vectorizer.vocabulary_)
        self.validation_data = self.word_vectorizer.transform(self.validation_data)
        self.validation_data = self.validation_data.todense()
        self.train_x_tensor = torch.from_numpy(np.array(self.training_data)).type(torch.FloatTensor)
        self.train_y_tensor = torch.from_numpy(np.array(self.training_labels)).long()
        self.validation_x_tensor = torch.from_numpy(np.array(self.validation_data)).type(torch.FloatTensor)
        self.validation_y_tensor = torch.from_numpy(np.array(self.validation_labels)).long()

        # Dataset
        self.training_dataset = TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.validation_dataset = TensorDataset(self.validation_x_tensor, self.validation_y_tensor)
        self.train_loader = DataLoader(self.training_dataset, batch_size=50, shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=50, shuffle=False)

        print("Done!")
        self.wipe()

    def save(self, name, dir="./"):
        # save dataset
        print(f'Dataset {name} is saved to {dir}')
        return torch.save(self, dir+name)

    def load(self, file):
        # load dataset
        print(f'Loading dataset {file}')
        return torch.load(file)

    def wipe(self):
        # Data and labels
        self.training_data = []
        self.validation_data = []
        self.training_labels = []
        self.validation_labels = []
        self.training_dataset = []
        self.validation_dataset = []

        # Tensors
        self.train_x_tensor = []
        self.train_y_tensor = []
        self.validation_x_tensor = []
        self.validation_y_tensor = []
        

    ###### PREPROCESS FUNCTIONS ######
    def preprocess_pandas(self, data, columns):
        df_ = pd.DataFrame(columns=columns)
        data['Sentence'] = data['Sentence'].str.lower()
        data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
        data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
        data['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')                                                       # remove special characters
        data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)                                                   # remove numbers
        for index, row in data.iterrows():
            word_tokens = word_tokenize(row['Sentence'])
            filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
            df_ = df_.append({
                "index": row['index'],
                "Class": row['Class'],
                "Sentence": " ".join(filtered_sent[0:])
            }, ignore_index=True)
        return data


###### CLASSIFICATION MODEL ######
class ClassyModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

        # Hyperparameters
        self.epochs = 1
        self.lr = 0.001
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x

    ###### VALIDATION FUNCTION ######
    def validate(self, validationLoader):
        with torch.no_grad():
            correct = 0
            total = 0
            
            for batch_nr, (data, labels) in enumerate(validationLoader):
                
                prediction = self(data)

                for i in range(len(labels)):
                    guess = torch.argmax(prediction[i], dim=-1)
                    correct += 1 if guess.item() == labels[i].item() else 0
                    total += 1
            
            return str(100*correct/total)[:4]+"%"

    def train(self, trainLoader, validationLoader):
        for epoch in range(self.epochs):
            # For each batch of data (since the dataset is too large to run all data through the network at once)
            for batch_nr, (data, labels) in enumerate(trainLoader):

                prediction = self(data)
                loss = self.loss_function(prediction, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                
                #Print the epoch, batch, and loss
                print(
                    f'\rEpoch {epoch+1} [{batch_nr+1}/{len(trainLoader)}] - Loss: {loss} - Acc.: {self.validate(validationLoader)}',
                    end=''
                )
    
    def saveModel(self, name, dir="./"):
        torch.save(self, dir+name)
        print(f'Model {name} is saved to {dir}')

    def saveState(self, name, dir="./"):
        torch.save(self.state_dict(), dir+name)
        print(f'State {name} is saved to {dir}')

    def loadModel(self, file):
        print(f'Model {file} is loaded.')
        return torch.load(file)

    def loadState(self, file):
        print(f'State {file} is loaded')
        return self.load_state_dict(torch.load(file))

######  FROM Tutorial https://github.com/python-engineer/pytorch-chatbot ######
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

        # Hyperparameters
        self.epochs = 1
        self.lr = 0.001
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

    def train(self, trainLoader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(self.epochs):
            for (words, labels) in trainLoader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)

                # Forward pass
                outputs = self(words)
                # if y would be one-hot, we must apply
                # labels = torch.max(labels, 1)[1]
                loss = self.loss_function(outputs, labels)

                # Backward and optimuze
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
        print(f'Final loss: {loss.item():.4f}')

    def saveModel(self, name, dir="./"):
        torch.save(self, dir+name)
        print(f'Model {name} is saved to {dir}')

    def saveState(self, name, dir="./"):
        torch.save(self.state_dict(), dir+name)
        print(f'State {name} is saved to {dir}')

    def loadModel(self, file):
        print(f'Model {file} is loaded.')
        return torch.load(file)

    def loadState(self, file):
        print(f'State {file} is loaded')
        return self.load_state_dict(torch.load(file)) 

    


##### IntentDataset #####
class IntentDataset(Dataset):
    def __init__(self):
        self.x_train = []
        self.y_train  = []
        self.all_words = []
        self.tags = []

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    
    def __len__(self):
        return len(self.x_train)

    
    def bow_preprocess(self, file):
        with open(file, 'r') as f:
            intents = json.load(f)
        
        all_words = []
        tags = []
        xy = []

        # loop through each sentence in our intents patterns
        for intent in intents['intents']:
            tag = intent['tag']
            # add to tag list
            tags.append(tag)
            for pattern in intent['patterns']:
                # tokenize each word in the sentence
                w = tokenize(pattern)
                # add to our word list
                all_words.extend(w)
                # add to xy pair
                xy.append((w, tag))
        # stem and lower each word
        ignore_words = ['?', ',', '!']
        all_words = [stem(w) for w in all_words if w not in ignore_words]
        # remove duplicates and sort
        all_words = sorted(set(all_words))
        tags = sorted(set(tags))

        print(len(xy), "patterns")
        print(len(tags), "tags:", tags)
        print(len(all_words), "unique stemmed words:", all_words)

        # create training data
        x_train = []
        y_train = []
        for (pattern_sentence, tag) in xy:
            # x: bag of words for each pattern_sentence
            bag = bag_of_words(pattern_sentence, all_words)
            x_train.append(bag)
            # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
            label = tags.index(tag)
            y_train.append(label)

        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.all_words = all_words
        self.tags = tags

    def trainLoader(self, batch_size=50, shuffle=True, num_workers = 0):
        return DataLoader(self, batch_size, shuffle=True)

    def save(self, name, dir="./"):
        print(f'Saving dataset {name} to {dir}')
        return torch.save(self, dir+name)

    def load(file):
        # load dataset
        print(f'Loading dataset {file}')
        return torch.load(file)

    

##### nltk_utils #####
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag