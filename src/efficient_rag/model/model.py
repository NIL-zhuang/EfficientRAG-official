from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    ContextPooler,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    StableDropout,
)


@dataclass
class SequenceTokenClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    sequence_logits: torch.FloatTensor = None
    token_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DebertaForSequenceTokenClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, token_labels: int = 2, sequence_labels: int = 3):
        super().__init__(config)

        self.token_labels = token_labels
        self.sequence_labels = sequence_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.sequence_classifier = nn.Linear(output_dim, sequence_labels)
        sequence_dropout = getattr(config, "cls_dropout", None)
        sequence_dropout = self.config.hidden_dropout_prob if sequence_dropout is None else sequence_dropout
        self.sequence_dropout = StableDropout(sequence_dropout)

        self.token_classifier = nn.Linear(config.hidden_size, token_labels)
        self.token_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.deberta.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        sequence_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]
        pooled_output = self.pooler(hidden_state)
        pooled_output = self.sequence_dropout(pooled_output)
        sequence_logits = self.sequence_classifier(pooled_output)

        token_output = self.token_dropout(hidden_state)
        token_logits = self.token_classifier(token_output)
        return SequenceTokenClassifierOutput(
            sequence_logits=sequence_logits,
            token_logits=token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
