import pdb
import math
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MultiHeadAttention
from recbole.model.loss import BPRLoss


class REDS(SequentialRecommender):
    def __init__(self, config, dataset, item_freq):
        super(REDS, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  
        self.dropout_prob = config["dropout_prob"]
        self.dropout_attn = config["dropout_attn"]
        self.topk = config["n_retrieval"]

        # load item frequency
        self.item_freq = item_freq.to(config['device'])

        # set embeddings
        self.item_freq = item_freq.to(config['device'])
        self.item_embedding_I = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding_I = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.freq_embedding_I = nn.Embedding(item_freq.max() + 1, self.hidden_size)

        self.item_embedding_C = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding_C = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.freq_embedding_C = nn.Embedding(item_freq.max() + 1, self.hidden_size)

        # define layers and loss 
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.trm_layers_IC = nn.ModuleList([
            trm_layer(
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                dropout=self.dropout_attn,
            ) for _ in range(self.n_layers)
        ])

        self.trm_layers_UC = nn.ModuleList([
            trm_layer(
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                dropout=self.dropout_attn,
            ) for _ in range(1)
        ])
        
        self.loss_type = config["loss_type"]
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, pos_items):
        extended_attention_mask = self.get_attention_mask(item_seq)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        ##Predictor UI##
        position_embedding_I = self.position_embedding_I(position_ids)
        freq_embedding_I = self.freq_embedding_I(self.item_freq[item_seq])

        item_emb_I = self.item_embedding_I(item_seq)
        item_emb_I = item_emb_I + position_embedding_I + freq_embedding_I
        item_emb_I = self.dropout(item_emb_I)
        item_emb_I = self.LayerNorm(item_emb_I)

        for i in range(self.n_layers):
            item_emb_I = self.trm_layers_IC[i](item_emb_I, item_emb_I, extended_attention_mask)
        output_UI = item_emb_I[:,-1,:]

        ##Predictor IC##
        position_embedding_C = self.position_embedding_C(position_ids)
        freq_embedding_C = self.freq_embedding_C(self.item_freq[item_seq])

        item_emb_C = self.item_embedding_C(item_seq)
        item_emb_C = item_emb_C + position_embedding_C + freq_embedding_C
        item_emb_C = self.dropout(item_emb_C)
        item_emb_C = self.LayerNorm(item_emb_C)

        pos_emb_C = self.item_embedding_C(pos_items)

        for i in range(self.n_layers):
            item_emb_C = self.trm_layers_IC[i](item_emb_C, item_emb_C, extended_attention_mask)
        output_IC = item_emb_C[:,-1,:]

        ##Retriever##
        topk = self.topk
        output_norm = F.normalize(output_IC, p=2, dim=1)
        pos_emb_norm = F.normalize(pos_emb_C, p=2, dim=1)
        similarity_matrix = torch.matmul(pos_emb_norm, output_norm.t()) #label serves as query
        similarity_matrix.fill_diagonal_(-float('inf'))
        _, topk_indices = torch.topk(similarity_matrix, topk, dim=1) #Topk U: (B, K, H)
        output_k = output_IC[topk_indices]

        ##Predictor UC##
        output_reshaped = output_IC.unsqueeze(1)
        output_concat = torch.cat((output_k, output_reshaped),dim=1)
        attention_mask_k = torch.zeros(output_reshaped.size(0), self.n_heads, topk+1, topk+1).to(output_IC.device)
        attn_state = self.trm_layers_UC[-1](output_reshaped, output_concat, attention_mask_k)
        output_UC = attn_state.reshape(-1, self.hidden_size) # w/ augmentation: (B*(K+1), H)

        return output_UI, output_IC, output_UC, topk_indices
                                                                                                                            
    def forward_eval(self, item_seq):
        extended_attention_mask = self.get_attention_mask(item_seq)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        ##Predictor UI##
        position_embedding_I = self.position_embedding_I(position_ids)
        freq_embedding_I = self.freq_embedding_I(self.item_freq[item_seq])

        item_emb_I = self.item_embedding_I(item_seq)
        item_emb_I = item_emb_I + position_embedding_I + freq_embedding_I
        item_emb_I = self.dropout(item_emb_I)
        item_emb_I = self.LayerNorm(item_emb_I)

        for i in range(self.n_layers):
            item_emb_I = self.trm_layers_IC[i](item_emb_I, item_emb_I, extended_attention_mask)
        output_UI = item_emb_I[:,-1,:]
        
        ##Predictor IC##
        position_embedding_C = self.position_embedding_C(position_ids)
        freq_embedding_C = self.freq_embedding_C(self.item_freq[item_seq])

        item_emb_C = self.item_embedding_C(item_seq)
        item_emb_C = item_emb_C + position_embedding_C + freq_embedding_C
        item_emb_C = self.dropout(item_emb_C)
        item_emb_C = self.LayerNorm(item_emb_C)

        for i in range(self.n_layers):
            item_emb_C = self.trm_layers_IC[i](item_emb_C, item_emb_C, extended_attention_mask)
        output_IC = item_emb_C[:,-1,:]
        
        ##Retriever##
        topk = self.topk
        output_norm = F.normalize(output_IC, p=2, dim=1)
        similarity_matrix = torch.matmul(output_norm, output_norm.t()) # target usesr rep serves as query
        similarity_matrix.fill_diagonal_(-float('inf'))
        _, topk_indices = torch.topk(similarity_matrix, topk, dim=1) #Topk U: (B, K, H)
        output_k = output_IC[topk_indices]

        ##Predictor UC##
        output_reshaped = output_IC.unsqueeze(1)
        output_concat = torch.cat((output_k, output_reshaped),dim=1)
        attention_mask_k = torch.zeros(output_reshaped.size(0), self.n_heads, 1, topk+1).to(output_IC.device)
        attn_state = self.trm_layers_UC[-1](output_reshaped, output_concat, attention_mask_k) 
        output_UC = attn_state[:,-1,:] # w/o augmentation: (B, H)
        
        return output_UI, output_IC, output_UC  

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        pos_items = interaction[self.POS_ITEM_ID]
        
        output_UI, output_IC, output_UC, topk_indices = self.forward(item_seq, pos_items)
        kpos_items = pos_items[topk_indices]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss

        else:  # self.loss_type = 'CE'
            test_item_emb_I = self.item_embedding_I.weight
            test_item_emb_C = self.item_embedding_C.weight

            logits_UI = torch.matmul(output_UI, test_item_emb_I.transpose(0,1))
            loss_UI = self.loss_fct(logits_UI, pos_items)

            logits_IC = torch.matmul(output_IC, test_item_emb_C.transpose(0, 1))
            logits_UC = torch.matmul(output_UC, test_item_emb_C.transpose(0, 1))
            loss_IC = self.loss_fct(logits_IC, pos_items)
            loss_UC = self.loss_fct(logits_UC, torch.cat((kpos_items, pos_items.unsqueeze(1)), dim=1).view(-1)) 

            loss = (loss_IC + loss_UC) + loss_UI
            
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        test_item = interaction[self.ITEM_ID]
        output_UI, output_IC, output_UC = self.forward_eval(item_seq)
        scores_UI = torch.matmul(output_UI, test_items_emb_I(test_item))
        scores_IC = torch.matmul(output_IC, test_items_emb_C(test_item))  
        scores_UC = torch.matmul(output_UC, test_items_emb_C(test_item))  
        scores = scores_UI + (scores_IC + scores_UC)

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        output_UI, output_IC, output_UC = self.forward_eval(item_seq)
        test_items_emb_I = self.item_embedding_I.weight
        test_items_emb_C = self.item_embedding_C.weight

        output_C = output_IC + output_UC

        combined_output = torch.cat([output_UI, output_C], dim=1)  # [B, 2h]
        combined_items = torch.cat([test_items_emb_I.transpose(0,1), test_items_emb_C.transpose(0,1)], dim=0)  # [2h, n]
        scores = torch.matmul(combined_output, combined_items)
 
        return scores

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, query, key, attention_mask):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(key)
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + query)
        return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, dropout):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
                n_heads, hidden_size, hidden_dropout_prob=dropout, attn_dropout_prob=dropout, layer_norm_eps=1e-12
        )

    def forward(self, hidden_states, k_hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, k_hidden_states, attention_mask)
        return attention_output

class trm_layer(nn.Module):
    def __init__(self, n_heads, hidden_size, dropout):
        super().__init__()
        self.layer = TransformerLayer(
            n_heads=n_heads,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = FeedForward(hidden_size=hidden_size, inner_size=hidden_size*4, dropout=dropout)

    def forward(self, input_tensor, key_tensor, attention_mask):
        hidden_states = self.layer(input_tensor, key_tensor,  attention_mask)
        output = self.ffn(hidden_states)

        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, inner_size)
        self.w_2 = nn.Linear(inner_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
