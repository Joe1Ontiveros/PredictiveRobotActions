# based off of https://www.youtube.com/watch?v=J_ksCv_r_rU
# tuned to my scenario
import os 
import torch 
import numpy as np 
import pandas as pd 
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LSTM(nn.Module):
    def __init__(self,input_len,hidden_size,num_class,num_layers):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_len,hidden_size,num_layers,batch_first=True)
        self.output_layer = nn.Linear(hidden_size,num_class) # linear ltsm for now

    def forward(self,X):
        # states of hidden and cells at 0
        hidden_states = torch.zeros(self.num_layers,X.size(0),self.hidden_size)
        cell_states = torch.zeros(self.num_layers,X.size(0),self.hidden_size)
        out,_= self.lstm(X,(hidden_states,cell_states))
        # reshape befer resubmitting to model 
        out = self.output_layer(out[:,-1,:])
        return out 
    

class LTSM_data:

    def __init__(self,pt):
     # a LTSM can decide/forget next input based on 
     # the incoming data flow 
        self.path = pt 
        self.data_pd = self.parse_csv(self.path)
        self.data_ld = self.data_loaders()
        
    def parse_csv(self, path):
        df = pd.read_csv(path)
        # Encode all gesture columns
        gesture_cols = ['gesture_1', 'gesture_2', 'gesture_3']
        for col in gesture_cols:
            df[col], _ = pd.factorize(df[col])
        X = df[gesture_cols].values.astype(np.float32)
        # Encode the label column
        y, uniques = pd.factorize(df['next_action'])
        y = y.astype(np.int64)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return {'inputs': X_tensor, 'labels': y_tensor}    
    def data_loaders(self, batch_size=12):
        # data_dict should be the output of parse_csv
        inputs = self.data_pd['inputs']
        labels = self.data_pd['labels']
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader
    
def trainer(num_epochs, model, lossf, opt: optim.Optimizer, traindt, nm_lyr, sq_ln):
    total_steps = len(traindt)
    print("STARTED TRAINING")
    for epoch in range(num_epochs):  # iterate for each EPOCH
        for batch, (inputs, labels) in enumerate(traindt):
            inputs = inputs.reshape(-1, 1, sq_ln)  # (batch, seq_len=1, input_len=3)
            labels = labels.long()  # CrossEntropyLoss expects class indices
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                model = model.to("cuda")
            output = model(inputs)
            loss = lossf(output, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Print every batch (for small datasets)
            print(f'EPOCH: {epoch}, BATCH: {batch} LOSS:{loss.item():>4f}')
    # Save the trained model
    torch.save(model.state_dict(), "trained_lstm_model.pth")
    print("Model saved to trained_lstm_model.pth")

def main():
    # hyper parameters: 
    batchsize = 12
    seq_len = 3
    hidden_size = 128 
    input_len = 3  # 3words/gestures
    num_layers = 2
    num_classes = 10         
    num_epochs = 6 
    learning_rate = 0.01
    model = LSTM(input_len, hidden_size, num_classes, num_layers)
    print(model)
    # loss func : 
    loss_func = nn.CrossEntropyLoss()
    optimz = optim.SGD(model.parameters(), lr=learning_rate)
    # Load data
    data = LTSM_data("datasets/action_predict/Balanced_Gesture_Command_Dataset.csv")  # replace with your actual CSV path
    train_loader = data.data_ld
    # Train
    trainer(num_epochs, model, loss_func, optimz, train_loader, num_layers, seq_len)

def build_vocab_from_csv(csv_path):
    """Build gesture and action vocabularies from the CSV for encoding/decoding."""
    df = pd.read_csv(csv_path)
    gesture_cols = ['gesture_1', 'gesture_2', 'gesture_3']
    gesture_set = set()
    for col in gesture_cols:
        gesture_set.update(df[col].unique())
    gesture_vocab = {g: i for i, g in enumerate(sorted(gesture_set))}
    action_vocab = list(pd.factorize(df['next_action'])[1])
    return gesture_vocab, action_vocab

def test_gesture_sequence(model, gesture_vocab, action_vocab, gestures):
    """
    Predict the next action given a list of 3 gesture strings.
    model: trained LSTM model
    gesture_vocab: dict mapping gesture string to int (from training)
    action_vocab: list of action strings (from training)
    gestures: list of 3 gesture strings
    """
    assert len(gestures) == 3, "You must provide exactly 3 gestures."
    # Encode gestures
    encoded = [gesture_vocab.get(g, -1) for g in gestures]
    if -1 in encoded:
        print("Unknown gesture in input:", gestures)
        return None
    X = torch.tensor(encoded, dtype=torch.float32).reshape(1, 1, 3)  # (batch=1, seq_len=1, input_len=3)
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            X = X.to("cuda")
            model = model.to("cuda")
        output = model(X)
        pred_idx = output.argmax(dim=1).item()
        pred_action = action_vocab[pred_idx]
        print(f"Predicted action for {gestures}: {pred_action}")
        return pred_action 

if __name__ == "__main__":
    # Train and save model
    main()
    # Load model for testing
    input_len = 3
    hidden_size = 128
    num_layers = 2
    num_classes = 10
    model = LSTM(input_len, hidden_size, num_classes, num_layers)
    model.load_state_dict(torch.load("trained_lstm_model.pth", map_location="cpu"))
    model.eval()
    # Prepare vocabularies
    csv_path = "datasets/action_predict/Balanced_Gesture_Command_Dataset.csv"
    gesture_vocab, action_vocab = build_vocab_from_csv(csv_path)
    # Example test
    test_gesture_sequence(
        model, gesture_vocab, action_vocab,
        ["thumb_left", "middle_finger", "one_right"]
    )