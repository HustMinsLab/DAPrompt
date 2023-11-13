# coding: UTF-8
import torch
import torch.nn as nn
from transformers import DebertaForMaskedLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
embedding_size = 768


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.deberta_model = DebertaForMaskedLM.from_pretrained(args.model_name).to(device)
        self.deberta_model2 = DebertaForMaskedLM.from_pretrained(args.model_name).to(device)
        self.deberta_model.resize_token_embeddings(args.vocab_size)
        self.deberta_model2.resize_token_embeddings(args.vocab_size)
        for param in self.deberta_model.parameters():
            param.requires_grad = True

        self.hidden_size = 768

        self.vocab_size = args.vocab_size

    def forward(self, batch_arg, arg_mask, mask_indices, batch_size):
        sent_emb = self.deberta_model.deberta(batch_arg, arg_mask)[0].to(device)

        event_pair_embed = torch.tensor([]).to(device)
        for i in range(batch_size):
            e_emb = self.extract_event(sent_emb[i], mask_indices[i])
            if i == 0:
                event_pair_embed = e_emb
            else:
                event_pair_embed = torch.cat((event_pair_embed, e_emb),dim=0)
        # event_pair_embed：batch×2×768
        mask1_predict=self.deberta_model.cls(event_pair_embed[:,0,:])
        mask2_predict=self.deberta_model2.cls(event_pair_embed[:,1,:])

        return mask1_predict,mask2_predict


    def extract_event(self, embed, mask_idx):
        mask_embed = torch.zeros(1, embedding_size).to(device)
        mask_embed = embed[mask_idx]
        mask_embed = torch.unsqueeze(mask_embed, 0)
        return mask_embed
    def handler(self, to_add, tokenizer):
        da = self.deberta_model.deberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp
