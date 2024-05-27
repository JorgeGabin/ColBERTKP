import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertConfig

from colbert.parameters import DEVICE


class KPEncoder(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "encoder"
    load_tf_weights = None

    def __init__(self, config: BertConfig, dim=128):
        super(KPEncoder, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

    def _init_weights(self, module):
        """ Initialise the weights (needed this for the inherited from_pretrained method to work) """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.bert.init_weights()
        self.linear.apply(self._init_weights)

    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(
            DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)
