from ChatbotBeta import *

import random
import json

bot_name = 'Bot'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### GOOD #####
with open('good_intent.json', 'r') as json_data:
    good_intents = json.load(json_data)

good_dataset = torch.load('./goodIntent_Dataset.pt')

good_words = good_dataset.all_words
good_tags = good_dataset.tags

good_model = torch.load('./goodIntent_Model.pt')
#good_model.eval()

##### BAD #####
with open('bad_intent.json', 'r') as json_data:
    bad_intents = json.load(json_data)

bad_dataset = torch.load('./badIntent_Dataset.pt')

bad_words = bad_dataset.all_words
bad_tags = bad_dataset.tags

bad_model = torch.load('./badIntent_Model.pt')

def main():
    model = torch.load("./IMDB_Model.pt")
    # model.eval()
    dataset =  torch.load("./IMDB_Dataset.pt")

    while True:
        response = requests.get("https://en.wikipedia.org/w/api.php?action=query&format=json&prop=info&list=&meta=&generator=random&inprop=url&grnnamespace=0")
        x = response.json()["query"]["pages"]
        for k in x.keys():   
            try: 
                title = x[k]["title"]
                url = x[k]["fullurl"]
            except:
                continue
                
        user_choice = input(f"{bot_name}: What do you think about {title}?\n url: {url} \n")
        print(f'You: {user_choice}')
        
        if user_choice == "exit":
            break
        validation_data = dataset.word_vectorizer.transform([user_choice])
        validation_data = validation_data.todense()
        validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
        prediction = model(validation_x_tensor)
        
        training_entry = torch.argmax(prediction[0])
        if training_entry == 1:
            positive()
        else:
            negative()
 
def positive():
    print(f"{bot_name}: I'm glad you enjoyed it. Can you tell me a bit more about what you liked?")
    while True:
        sentence = input("You: ")

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, good_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = good_model(X)
        _, predicted = torch.max(output, dim=1)

        tag = good_tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        
        if prob.item() > 0.75:
            for intent in good_intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")

        if tag == 'EOS':
            break
            
def negative():
    print(f"{bot_name}: Can you tell me a bit more about what you didn't like?")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, bad_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = bad_model(X)
        _, predicted = torch.max(output, dim=1)

        tag = bad_tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        
        if prob.item() > 0.75:
            for intent in bad_intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")

        if tag == 'EOS':
            break
            
if __name__ == "__main__":
    main()                
