from ChatbotBeta import *

def main():
    model = torch.load("./IMDB_Model.pt")
    # model.eval()
    dataset =  ChatbotDataset.load("./IMDB_Dataset.pt")

    while True:
        response = requests.get("https://en.wikipedia.org/w/api.php?action=query&format=json&prop=info&list=&meta=&generator=random&inprop=url&grnnamespace=0")
        x = response.json()["query"]["pages"]
        for k in x.keys():   
            try: 
                title = x[k]["title"]
                url = x[k]["fullurl"]
            except:
                continue
                
        user_choice = input(f'Chatbot: What do you think about {title}?\n url: {url} \n')
        print(f'You: {user_choice}')
        
        if user_choice == "exit":
            break
        validation_data = dataset.word_vectorizer.transform([user_choice])
        validation_data = validation_data.todense()
        validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
        prediction = model(validation_x_tensor)
        
        training_entry = torch.argmax(prediction[0])
        if training_entry == 1:
            print("Chatbot: oh you think its good")
        else:
            print("Chatbot: you think its bad")
                
if __name__ == "__main__":
    main()                