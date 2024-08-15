import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.templates = self.dataframe['template'].values
        self.inputs = self.dataframe['input'].values
        self.labels = self.dataframe['label'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        template = self.templates[idx]
        input_text = self.inputs[idx]
        label = self.labels[idx]
        return template, input_text, label

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['input'] = df['input'].apply(lambda x: x.lower().strip())
    df['template'] = df['template'].apply(lambda x: x.lower().strip())
    return df

data_path = 'dummy_data.csv'
raw_data = preprocess_data(data_path)
dataset = NewsDataset(raw_data)

# Splitting into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



class NeuralTemplateSelectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralTemplateSelectionModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_text):
        embedded = self.embedding(input_text)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        output = self.softmax(output)
        return output

input_size = 10000  # Dummy value
hidden_size = 128
output_size = len(set(raw_data['template'].values))  # Number of unique templates

model = NeuralTemplateSelectionModel(input_size, hidden_size, output_size)




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for templates, inputs, labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

   
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for templates, inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_preds.append(torch.argmax(outputs, dim=1).numpy())
            val_labels.append(labels.numpy())

    val_accuracy = accuracy_score(np.concatenate(val_labels), np.concatenate(val_preds))
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")


model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for templates, inputs, labels in test_loader:
        outputs = model(inputs)
        test_preds.append(torch.argmax(outputs, dim=1).numpy())
        test_labels.append(labels.numpy())

test_accuracy = accuracy_score(np.concatenate(test_labels), np.concatenate(test_preds))
print(f"Test Accuracy: {test_accuracy:.4f}")



model_path = 'neural_template_selection_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


def fill_template(template, slots):
    """
    Replace placeholders in the template with actual data from the slots.
    """
    for slot, value in slots.items():
        template = template.replace(f"{{{{{slot}}}}}", value)
    return template

with open("../../templates.json", "r") as file:
    templates = file.read()


with open("../../slots.json", "w") as file:
    slots = file.read()
    

def infer_and_generate_script(model, input_text, templates, slots):
    model.eval()
    input_tensor = torch.tensor([input_text]) 
    output = model(input_tensor)
    predicted_template_idx = torch.argmax(output, dim=1).item()
    
    selected_template = templates[predicted_template_idx]
    generated_script = fill_template(selected_template, slots)
    
    return generated_script

sample_input = "sample input text"
generated_script = infer_and_generate_script(model, sample_input, templates, slots)

script_path = 'generated_news_script.txt'
with open(script_path, 'w') as file:
    file.write(generated_script)

