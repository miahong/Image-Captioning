import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1] # delete the last fc
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.init_weights()
    
    def init_weights(self):
        self.embed.weight.data.normal_(0.0,0.02)
        self.embed.bias.data.fill_(0)
            
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):  
        self.embed.weight.data.uniform_(-0.1,0.1)
        self.fc.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
        
    def forward(self, features, captions):
        captions = captions[:, :-1]
        captions = self.embed(captions)
        # Concatenate the feature vectors for image and captions.
        # Features shape : (batch_size, embed_size)
        # Word embeddings shape : (batch_size, caption length , embed_size)
        # output shape : (batch_size, caption length, embed_size)
        
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        outputs, _ = self.lstm(inputs)
        outputs = self.fc(outputs)
        return outputs
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        preds =[]
        
        for i in range(max_len):
            outputs, states = self.lstm(inputs, states)
            outputs = self.fc(outputs.squeeze(1))
            word_index = outputs.max(1)[1]
            
            preds.append(word_index.item())
            if (word_index == 1):
                break
            inputs = self.embed(word_index).unsqueeze(1)
            
        return preds
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        