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
        outputs = self.visual_model(pixel_values)
        print(pixel_values.shape, outputs.shape)
        encoder_outputs = outputs.pooler_output
        print(encoder_outputs.shape)
        visual_embs = self.visual_embeddings(encoder_outputs)
        print("AAAA", visual_embs.shape)
        visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], self.args.n_visual_tokens, -1))
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
        visual_embs = self.get_visual_embs(torch.cat(images, dim=0)).reshape(len(images), images[0].shape[0], self.args.n_visual_tokens, -1)
        for i in range(input_embs.shape[0]):
            for j in range(len(images)):
                if image_ranges[j][0] == image_ranges[j][1]:
                    continue
                input_embs[i][image_ranges[j][0]:image_ranges[j][1]] = visual_embs[i][j]

        output = self.lm(input_embeddings=input_embs,
                        attention_mask=attention_mask,
                        labels=labels)
