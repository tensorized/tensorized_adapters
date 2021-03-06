from typing import NamedTuple, Union, Callable
import logging
import torch
from torch_nn_layers import CPLayer, TRLayer, TTTLayer, TuckerLayer, MFLayer
import torch.nn as nn

from transformers import BertModel
from transformers.models.roberta.modeling_roberta import ACT2FN, RobertaSelfOutput
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import ACT2FN, DebertaV2SelfOutput

SELF_OUTPUT = {
    "bert-base-cased": BertSelfOutput,
    "roberta-base": RobertaSelfOutput,
    "microsoft/deberta-v3-base": DebertaV2SelfOutput

}


from transformers import AutoModelForSequenceClassification

import torch.nn as nn


class AdapterConfig(NamedTuple):
    adapter_projection_type: str
    model_type: str
    hidden_size: int
    adapter_size: int
    adapter_act: Union[str, Callable]
    adapter_initializer_range: float
    tensorized_layer_rank: int
    if_adapter_like : bool
    


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def freeze_all_parameters(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    return model


logging.basicConfig(level=logging.INFO)


class LinearizedAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(LinearizedAdapter, self).__init__()

        
        self.proj_type = config.adapter_projection_type
        self.config = config
        self.tensorized_layer_rank = config.tensorized_layer_rank
        self.if_adapter_like = config.if_adapter_like
        
        # print(f"\n\n\n\n\n\n\n\n\n\n\n\n {config.hidden_size, self.proj_type, PROJECTION_TYPES[self.proj_type]} \n\n\n\n\n\n\n\n\n\n\n\n")

        if self.if_adapter_like:
            # print(f"\n\n\n\n\n\n\n\n\n\n\n\n 'adapterized' \n\n\n\n\n\n\n\n\n\n\n\n")
            self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
            nn.init.normal_(self.down_project.weight, std=config.adapter_initializer_range)
            nn.init.zeros_(self.down_project.bias)
            
            
            self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
            nn.init.normal_(self.up_project.weight, std=config.adapter_initializer_range)
            nn.init.zeros_(self.up_project.bias)
            
        else:
            # print(f"\n\n\n\n\n\n\n\n\n\n\n\n 'not adapterized' \n\n\n\n\n\n\n\n\n\n\n\n")
            self.tensorized_layer = nn.Linear(config.hidden_size , config.hidden_size)
            nn.init.normal_(self.tensorized_layer.weight, std=config.adapter_initializer_range)
            nn.init.zeros_(self.tensorized_layer.bias)
            
        if isinstance(config.adapter_act, str):
            self.activation = ACT2FN[config.adapter_act]
        else:
            self.activation = config.adapter_act

    
    def forward(self, hidden_states: torch.Tensor):

        if self.if_adapter_like:
            down_projected = self.down_project(hidden_states)
            activated = self.activation(down_projected)
            tensorized_projection = self.up_project(activated)
        else:
            tensorized_projection = self.tensorized_layer(hidden_states)
            tensorized_projection = self.activation(tensorized_projection) # do you need activation?
        
        return hidden_states + tensorized_projection

class BertAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: BertSelfOutput,
                 config: AdapterConfig):
        super(BertAdaptedSelfOutput, self).__init__()
        self.config = config
        self.self_output = self_output
        # print(SELF_OUTPUT[config.model_type])
        # self.self_output = SELF_OUTPUT[config.model_type]
        self.adapter = LinearizedAdapter(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class RobertaAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: RobertaSelfOutput,
                 config: AdapterConfig):
        super(RobertaAdaptedSelfOutput, self).__init__()
        self.config = config
        self.self_output = self_output
        # print(SELF_OUTPUT[config.model_type])
        # self.self_output = SELF_OUTPUT[config.model_type]
        self.adapter = LinearizedAdapter(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    
class DebertaAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: DebertaV2SelfOutput,
                 config: AdapterConfig):
        super(DebertaAdaptedSelfOutput, self).__init__()
        self.config = config
        self.self_output = self_output
        # print(SELF_OUTPUT[config.model_type])
        # self.self_output = SELF_OUTPUT[config.model_type]
        self.adapter = LinearizedAdapter(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def adapt_output_bert(config: AdapterConfig):
    return lambda self_output: BertAdaptedSelfOutput(self_output, config=config)

def adapt_output_roberta(config: AdapterConfig):
    return lambda self_output: RobertaAdaptedSelfOutput(self_output, config=config)

def adapt_output_deberta(config: AdapterConfig):
    return lambda self_output: DebertaAdaptedSelfOutput(self_output, config=config)


def add_tensorized_adapters(model: nn.Module, config: AdapterConfig) -> nn.Module:
    if config.model_type in ['bert-base-cased','roberta-base', 'microsoft/deberta-v3-base']:
        for layer in model.encoder.layer:
            if config.model_type=='bert-base-cased':
                layer.attention.output = adapt_output_bert(config)(layer.attention.output)
                layer.output = adapt_output_bert(config)(layer.output)
            if config.model_type=='roberta-base':
                layer.attention.output = adapt_output_roberta(config)(layer.attention.output)
                layer.output = adapt_output_roberta(config)(layer.output)
            if config.model_type=='microsoft/deberta-v3-base':
                layer.attention.output = adapt_output_deberta(config)(layer.attention.output)
                layer.output = adapt_output_deberta(config)(layer.output)                
    return model


def unfreeze_bert_adapters(model: nn.Module) -> nn.Module:
    # Unfreeze trainable parts ??? layer norms and adapters
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, (LinearizedAdapter, nn.LayerNorm)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return model


def add_model_specific_adapters(model, adapter_config):
    if adapter_config.model_type=='roberta-base':
        model.roberta = add_tensorized_adapters(model.roberta, adapter_config)
        model.roberta = freeze_all_parameters(model.roberta)
        model.roberta = unfreeze_bert_adapters(model.roberta)
        model.classifier.requires_grad = True
    elif adapter_config.model_type=='bert-base-cased':
        model.bert = add_tensorized_adapters(model.bert, adapter_config)
        model.bert = freeze_all_parameters(model.bert)
        model.bert = unfreeze_bert_adapters(model.bert)
        model.classifier.requires_grad = True
    elif adapter_config.model_type=='microsoft/deberta-v3-base':
        model.deberta = add_tensorized_adapters(model.deberta, adapter_config)
        model.deberta = freeze_all_parameters(model.deberta)
        model.deberta = unfreeze_bert_adapters(model.deberta)
        model.classifier.requires_grad = True
        
    return model
