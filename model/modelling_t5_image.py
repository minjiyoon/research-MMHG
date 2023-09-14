import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    CLIPVisionModel
)

class T5Image(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()

        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
        self.visual_model = CLIPVisionModel.from_pretrained(args.visual_model)

        self.tokenizer = tokenizer
        self.args = args

        self.input_embeddings = self.lm.get_input_embeddings()
        hidden_size = self.visual_model.config.hidden_size
        embedding_dim = self.input_embeddings.embedding_dim * self.args.n_visual_tokens
        self.visual_embeddings = nn.Linear(hidden_size, embedding_dim)

        if self.args.freeze_lm:
            self.lm.eval()
            print("Freezing the LM.")
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm.train()

        self.visual_model.eval()
        for param in self.visual_model.parameters():
            param.requires_grad = False

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        outputs = self.visual_model(torch.reshape(pixel_values, (-1, pixel_values.shape[-3], pixel_values.shape[-2], pixel_values.shape[-1])))
        encoder_outputs = outputs.pooler_output
        visual_embs = self.visual_embeddings(encoder_outputs)
        visual_embs = torch.reshape(visual_embs, (pixel_values.shape[0], pixel_values.shape[1],  self.args.n_visual_tokens, -1))
        return visual_embs

    def train(self, mode=True):
        super(T5Image, self).train(mode=mode)
        # Overwrite train() to ensure frozen models remain frozen.
        if self.args.freeze_lm:
            self.lm.eval()
        self.visual_model.eval()


    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        images=None,
        image_ranges=None,
    ):
        if images == None and image_ranges == None:
            print('*** You didnt give images... ***')
            return self.lm(input_ids, attention_mask, labels)

        input_embs = self.input_embeddings(input_ids)
        visual_embs = self.get_visual_embs(images)
        for i in range(visual_embs.shape[0]):
            for j in range(visual_embs.shape[1]):
                if image_ranges[i][j][0] == image_ranges[i][j][1]:
                    continue
                input_embs[i][image_ranges[i][j][0]:image_ranges[i][j][1]] = visual_embs[i][j]

        output = self.lm(inputs_embeds=input_embs,
                        attention_mask=attention_mask,
                        labels=labels)
