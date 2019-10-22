import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):

    def __init__(self, embedding, params):

        super(CNN_Text, self).__init__()

        self.args = params
        n_words = embedding.shape[0]
        emb_dim = embedding.shape[1]
        filters = params.filters

        if params.kernels <= 3:
            Ks = [i for i in range(1, params.kernels + 1)]
        else:
            Ks = [i for i in range(1, params.kernels + 1, 2)]

        self.embed = nn.Embedding(n_words, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding), requires_grad=False)

        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K) for K in Ks])

        self.fc = nn.Linear(len(Ks) * filters, 1)

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.embed_dropout = nn.Dropout(p=self.args.embed_dropout)

    def fc_layer(self, x, layer):
        x = layer(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        return x

    def encoder(self, x):
        x = self.embed(x)
        x = x.transpose(1, 2)
        x = self.embed_dropout(x)

        h = [self.fc_layer(x, self.convs[i]) for i in range(len(self.convs))]

        return h

    def forward(self, Note):
        h = self.encoder(Note)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]

        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)

class CNN_Text_Attn(nn.Module):
    def __init__(self, embedding, params):

        super(CNN_Text_Attn, self).__init__()

        self.args = params
        n_words = embedding.shape[0]
        emb_dim = embedding.shape[1]
        filters = params.filters

        if (params.kernels <= 3):
            Ks = [i for i in range(1, params.kernels + 1)]
        else:
            Ks = [i for i in range(1, params.kernels + 1, 2)]

        self.embed = nn.Embedding(n_words, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(embedding), requires_grad=False)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K) for K in Ks])
        self.U = nn.Linear(filters, 1, bias=False)

        self.fc = nn.Linear(filters, 1)

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.embed_dropout = nn.Dropout(p=self.args.embed_dropout)
        self.padding = nn.ModuleList([nn.ConstantPad1d((0, K - 1), 0) for K in Ks])


    def fc_layer(self, x, layer, padding):
        x = layer(padding(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)

        return x

    def encoder(self, x):
        x = self.embed(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)

        h = [self.fc_layer(x, self.convs[i], self.padding[i]) for i in range(len(self.convs))]

        return h

    def forward(self, Note, interpret=False):

        text = Note[0]
        attn_mask = torch.cat([Note[1]] * len(self.convs), 1)
        h = self.encoder(text)

        h = torch.cat(h, 2)

        alpha = F.softmax(torch.add(self.U(h.transpose(1, 2) / (self.args.filters)**0.5), attn_mask.unsqueeze(2)), dim=1)

        h = h.matmul(alpha).squeeze(2)
        h = self.dropout(h)
        y_hat = self.fc(h)

        if interpret:
            return torch.sigmoid(y_hat), [alpha.squeeze()]
        else:
            return torch.sigmoid(y_hat)
