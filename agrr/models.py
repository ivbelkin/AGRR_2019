import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel


class BertAgrrModel(PreTrainedBertModel):

    def __init__(self, config):
        super(BertAgrrModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.sentence_classifier = nn.Linear(config.hidden_size, 2)
        self.full_annotation_classifier = nn.Linear(config.hidden_size, 6)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                labels=None, gaps=None, tags=None):
        sequence_output, pooled_output = \
            self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output, pooled_output = \
            self.dropout(sequence_output), self.dropout(pooled_output)
        sentence_logits = self.sentence_classifier(pooled_output)
        full_annotation_logits = self.full_annotation_classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            # sentence sclasification loss
            sentence_loss = loss_fct(sentence_logits.view(-1, 2), labels.view(-1))

            # full annotation loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = full_annotation_logits.view(-1, 6)[active_loss]
                active_labels = tags.view(-1)[active_loss]
                full_annotation_loss = loss_fct(active_logits, active_labels)
            else:
                full_annotation_loss = loss_fct(full_annotation_logits.view(-1, 6), tags.view(-1))

            loss = sentence_loss + full_annotation_loss
            return loss
        else:
            sentence_logits = torch.log_softmax(sentence_logits, dim=1)
            full_annotation_logits = torch.log_softmax(full_annotation_logits, dim=2)
            return sentence_logits, full_annotation_logits, full_annotation_logits
