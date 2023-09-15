class PEFT(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.lm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=self.config)

        if self.args.freeze_lm:
            self.lm.eval()
            print("Freezing the LM.")
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm.train()

        self.input_embeddings = self.lm.get_input_embeddings()

        hidden_size = self.lm.config.hidden_size
        config = AutoConfig.from_pretrained(args.text_model)
        self.text_model = RobertaModel.from_pretrained(args.text_model, config=config)
        self.text_pooler = TextPooler(config)
        self.text_embeddings = nn.Linear(self.text_model.config.hidden_size, hidden_size)
        if args.text_position_type != "none":
            self.text_position_embeddings = nn.Embedding(args.max_output_length + 1, hidden_size) # + 1 for padding neighbors

        self.text_model.eval()
        for name, param in self.text_model.named_parameters():
            param.requires_grad = False

        #self.visual_model = CLIPVisionModel.from_pretrained(args.visual_model)
        #self.visual_embeddings = nn.Linear(self.visual_model.config.hidden_size, hidden_size)
        #self.visual_model.eval()

        #for param in self.visual_model.parameters():
        #    param.requires_grad = False

    def get_text_embs(self, input_ids, attention_mask, pos_ids=None):
        batch_size, neighbor_num, seq_len = input_ids.shape
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)

        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        encoder_outputs = self.text_pooler(outputs.last_hidden_state)
        text_embs = self.text_embeddings(encoder_outputs)
        text_embs = text_embs.reshape(batch_size, neighbor_num, -1)

        if pos_ids is None:
            return text_embs
        else:
            return text_embs + self.text_position_embeddings(pos_ids)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        outputs = self.visual_model(pixel_values)
        encoder_outputs = outputs.pooler_output
        visual_embs = self.visual_embeddings(encoder_outputs)
        return visual_embs

    def train(self, mode=True):
        super(PEFT, self).train(mode=mode)
        # Overwrite train() to ensure frozen models remain frozen.
        if self.args.freeze_lm:
            self.lm.eval()
        self.text_model.eval()
        #self.visual_model.eval()

    def forward(self, input_ids, attention_mask, labels, neighbor_input_ids=None, neighbor_attention_mask=None, neighbor_pos_ids=None):
        if neighbor_input_ids is None:
            neighbor_embeds = None
            neighbor_attention_mask = None
        else:
            neighbor_embeds = self.get_text_embs(neighbor_input_ids, neighbor_attention_mask, neighbor_pos_ids)
            neighbor_attention_mask = neighbor_attention_mask[:, :, 0]

        if self.args.lora_type == "gill":
            input_embs = self.input_embeddings(input_ids)
            input_embs = torch.cat((neighbor_embs, input_embeds), dim=1)
            attention_mask = torch.cat((neighbor_attention_mask, attention_mask), dim=1)
            labels = torch.cat((-100 * torch.ones_like(neighbor_embs), labels), dim=1)
            output = self.lm(input_embs=input_embs, attention_mask=attention_mask, labels=labels)

        return output
