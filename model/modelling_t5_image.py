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

        embedding_dim = self.input_embeddings.embedding_dim * self.args.n_visual_tokens
        self.visual_embeddings = nn.Linear(hidden_size, embedding_dim)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        outputs = self.visual_model(pixel_values)
        encoder_outputs = outputs.pooler_output
        visual_embs = self.visual_embeddings(encoder_outputs)
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
        intput_ids,
        attention_mask,
        labels,
        images=None,
        text_lens=None,
    ):
        if images == None and text_lens == None:
            print('*** You didnt give images... ***')
            return self.lm(input_ids, attention_mask, labels)

        input_embs = self.input_embeddings(labels)
        visual_embs = self.get_visual_embs(images)

        for i in range(input_embs.shape[0]):
            if visual_embs.shape[0] + text_lens[i] > input_embs.shape[0]:
                visual_embs = visual_embs[:(input_embs.shape[0] - text_lens[i])]
            input_embs[i][text_lens[i]:] = visual_embs

        output = self.lm(input_embeddings=input_embs,
                        attention_mask=full_attention_mask,
                        labels=labels)
