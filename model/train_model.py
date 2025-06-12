import torch
import torchtext

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def yield_tokens(data):
    for _,line in data:
        yield tokenizer(line)

def prepare_data():

    # Workflow:
    # 1. Load the IMDb dataset (only train split)
    # 2. Tokenize each sentence (use 'basic_english' tokenizer)
    # 3. Build vocab from all tokens
    # 4. Define two pipelines:
    #    - text_pipeline(text) -> token IDs
    #    - label_pipeline(label) -> 0/1
    # 5. Pad and collate into batches
    # 6. Return DataLoader and vocab

    # Hints:
    # Use torchtext.datasets.IMDB(split='train')
    # Use torchtext.data.utils.get_tokenizer
    # Use torchtext.vocab.build_vocab_from_iterator
    # Use torch.nn.utils.rnn.pad_sequence

    train_data = list(torchtext.datasets.IMDB(split='train'))

    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_data),specials=["<pad>","<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: 1 if x == 'pos' else 0

    def collate_batch(data):
        text_data, label_data = [],[]
        for label,text in data:
            text_data.append(torch.tensor(text_pipeline(text),dtype=torch.int64))
            label_data.append(torch.tensor(label_pipeline(label),dtype=torch.int64))
        text_data = torch.nn.utils.rnn.pad_sequence(text_data,batch_first=True,padding_value=vocab["<pad>"])
        return text_data.to(device),torch.tensor(label_data).to(device)

    dataloader = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True,collate_fn=collate_batch)

    return dataloader,vocab

class SentimentModel(torch.nn.Module):
    def __init__(self,vocab_size,embedding_dim,padding_idx,hidden_size,out_features):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size,embedding_dim,padding_idx)
        self.lstm = torch.nn.LSTM(embedding_dim,hidden_size,batch_first=True)
        self.fc = torch.nn.Linear(hidden_size,out_features)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self,text):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))    # [batch_size, seq_len, embed_dim]
        _,(hidden,_) = self.lstm(embedded)                 # hidden: [1, batch_size, hidden_dim]
        hidden = hidden.squeeze(0)                       # [batch_size, hidden_dim]
        output = self.fc(hidden)                         # [batch_size, output_dim]
        return output


def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for texts,labels in dataloader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/ len(dataloader)

if __name__ == '__main__':
    dataloader, vocab = prepare_data()
    VOCAB_SIZE = len(vocab)   # vocab size
    EMBED_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = 2       # binary classification (pos/neg)
    PADDING_IDX = vocab["<pad>"]

    model = SentimentModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBED_DIM, hidden_size=HIDDEN_DIM, out_features=OUTPUT_DIM, padding_idx=PADDING_IDX).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    loss = train_model(model, dataloader, optimizer, criterion)
    print(f"Training loss: {loss:.4f}")