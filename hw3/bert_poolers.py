import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel, BertModel,
    BertEmbeddings, BertEncoder, BertForSequenceClassification, BertPooler,
)

class MeanMaxTokensBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.activation = nn.Tanh()
        # raise NotImplementedError

    def forward(self, hidden_states, *args, **kwargs):
        # hidden_states (batch_size, token_size, hidden_size)
        mean_part = torch.sum(hidden_states, 1) # (batch_size, hidden_size)
        max_part, _ = torch.max(hidden_states, 1) # (batch_size, hidden_size)
        c_mmt = torch.cat((mean_part, max_part), 1) # (batch_size, hidden_size*2)
        pooled_output = self.linear(c_mmt) # (batch_size, hidden_size)
        pooled_output = self.activation(pooled_output) # (batch_size, hidden_size)
        return pooled_output

class MyBertPooler_Stochastic(nn.Module):
    def __init__(self, config, sample):
        super().__init__()
        self.sample = sample # # of sampling from multinomial distribution
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.ReLU = nn.ReLU()
        self.activation = nn.Tanh()
        # raise NotImplementedError

    # multinomial_sample(self, t: torch, i: # of sampling) 
    # (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size), 
    # features pooled based on multinomial distribution, sampled i times with redundancy
    def multinomial_sample(self, t, i):
        batch_size = t.size()[0]
        seq_len = t.size()[1]
        t = t.transpose(1,2).reshape(-1, seq_len)        
        mask = torch.eq(torch.sum(t,1),0).view(-1,1)
        t = torch.add(t,mask)
        mult = torch.multinomial(t,i)
        t = torch.gather(t, -1, mult)
        t = torch.add(t, torch.mul(mask,-1))
        t = t.sum(1).view(batch_size, -1)
        return t  

    def forward(self, hidden_states, *args, **kwargs): # hidden_states (batch_size, seq_len, hidden_size)
        batch_size = hidden_states.size()[0]
        representation = self.ReLU(hidden_states) # ReLU representation (batch_size, seq_len, hidden_size)
        pooled_output = self.multinomial_sample(representation, self.sample) # pooled_output (batch_size, hidden_size)
        #print(pooled_output.size())
        pooled_output = self.fc(pooled_output) # (batch_size, hidden_size)
        pooled_output = self.activation(pooled_output) # (batch_size, hidden_size)
        return pooled_output

class MyBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError

    def forward(self, hidden_states, *args, **kwargs):
        raise NotImplementedError


class MyBertConfig(BertConfig):
    def __init__(self, pooling_layer_type="CLS", **kwargs):
        super().__init__(**kwargs)
        self.pooling_layer_type = pooling_layer_type


class MyBertModel(BertModel):

    def __init__(self, config: MyBertConfig):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        if config.pooling_layer_type == "CLS":
            # See src/transformers/models/bert/modeling_bert.py#L610
            # at huggingface/transformers (9f43a425fe89cfc0e9b9aa7abd7dd44bcaccd79a)
            self.pooler = BertPooler(config)
        elif config.pooling_layer_type == "MEAN_MAX":
            self.pooler = MeanMaxTokensBertPooler(config)
        elif config.pooling_layer_type == "MINE": # MY_STO1
            self.pooler = MyBertPooler_Stochastic(config, 1)
        elif config.pooling_layer_type == "MY_STO2":
            self.pooler = MyBertPooler_Stochastic(config, 2)
        elif config.pooling_layer_type == "MY_STO3":
            self.pooler = MyBertPooler_Stochastic(config, 3)
        elif config.pooling_layer_type == "MY_STO4":
            self.pooler = MyBertPooler_Stochastic(config, 4)
        elif config.pooling_layer_type == "MY_STO5":
            self.pooler = MyBertPooler_Stochastic(config, 5)
        elif config.pooling_layer_type == "MY_STO6":
            self.pooler = MyBertPooler_Stochastic(config, 6)
        elif config.pooling_layer_type == "MY_STO7":
            self.pooler = MyBertPooler_Stochastic(config, 7)
        elif config.pooling_layer_type == "MY_STO8":
            self.pooler = MyBertPooler_Stochastic(config, 8)
        else:
            raise ValueError(f"Wrong pooling_layer_type: {config.pooling_layer_type}")

        self.init_weights()

    @property
    def pooling_layer_type(self):
        return self.config.pooling_layer_type


class MyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = MyBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if self.bert.pooling_layer_type in ["CLS", "MEAN_MAX"]:
            return super().forward(
                input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, labels, output_attentions, output_hidden_states, return_dict
            )
        else:
            return super().forward(
                input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, labels, output_attentions, output_hidden_states, return_dict
            )
