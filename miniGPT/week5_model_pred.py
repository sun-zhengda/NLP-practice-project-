import torch
import torch.nn.functional as F
from week5_exercise import TransDecoderLayers

def tokenize(text,char2idx):
    return torch.tensor([char2idx[char] for char in text],dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')


def post_treatment(tokenized_text,model,idx2char):
    y_hat = model(tokenized_text.reshape(1,-1))
    prob = F.softmax(y_hat,dim=-1)
    max_token = prob.argmax(dim=-1).reshape(-1)
    return idx2char[int(max_token[-1])]


def main(text,outputlen=4):
    split = outputlen
    model_details = torch.load('miniGPTbaseline.bin',weights_only=False)
    hyperparams = model_details['hyperparams']
    char2idx = model_details['char2idx']
    idx2char = model_details['idx2char']
    vocab_size = len(char2idx)
    model = TransDecoderLayers(hyperparams['embedding_dim'],
                               hyperparams['position_dim'],
                               hyperparams['ffn_dim'],
                               hyperparams['nhead'],
                               vocab_size,
                               hyperparams['num_layers'],
                               hyperparams['dropout']).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(model_details['model_dict'])
    model.eval()
    while outputlen:
        with torch.no_grad():
            input = tokenize(text,char2idx)
            output = post_treatment(input,model,idx2char)
        text += output
        outputlen -= 1
    return text[-split:]

print(main('美元'))
 
