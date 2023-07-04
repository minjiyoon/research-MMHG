import argparse

import torch
import copy
import warnings
from transformers import EncoderDecoderModel, AutoTokenizer

warnings.filterwarnings("ignore")

MODEL_DIR = 'model'

LAYOUTS = {
    's1': 'SD|SD|SD|SD|SD|SD',
    's2': 'S|SD|D|S|SD|D|S|SD|D',
    'p1': 'S|SD|S|SD|S|SD|S|SD',
    'p2': 'S|S|SD|S|S|SD|S|S|SD',
    'e1': 'SD|SD|SD|S|S|S|S|S|S',
    'e2': 'S|SD|D|S|SD|D|S|S|S|S',
    'l1': 'S|S|S|S|S|S|SD|SD|SD',
    'l2': 'S|S|S|S|S|SD|D|S|SD|D',
    'b1': 'S|S|SD|D|S|SD|D|S|S|S',
    'b2': 'S|S|SD|SD|SD|S|S|S|S',
    'f12': 'S|S|S|S|S|S|S|S|S|S|S|S',
    'f8': 'S|S|S|S|S|S|S|S',
    'f6': 'S|S|S|S|S|S',
}


def prepare_decoder():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--layout', default='s1', choices=['s1', 's2', 'p1', 'p2', 'e1', 'e2',
                                                           'l1', 'l2', 'b1', 'b2', 'f12', 'f8', 'f6'],
                        help='S|D encoders layout')
    parser.add_argument('--max_length', default=128)
    parser.add_argument('--max_neighbors', default=64)
    config = parser.parse_args()

    MAX_LENGTH = int(config.max_length)
    MAX_NEIGHBORS = int(config.max_neighbors)
    ENCODER_LAYOUT = {}
    for idx, block_pattern in enumerate(LAYOUTS[config.layout].split('|')):
        ENCODER_LAYOUT[str(idx)] = {"attention": True if 'S' in block_pattern else False,
                                    "crossattention": True if 'D' in block_pattern else False}
    NUM_HIDDEN_LAYERS = len(ENCODER_LAYOUT.keys())

    # load pre-trained bert model and tokenizer
    BERT_CHECKPOINT = f'patrickvonplaten/bert2bert_cnn_daily_mail'
    tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT)
    bert_model = EncoderDecoderModel.from_pretrained(BERT_CHECKPOINT)

    # load dummy config and change specifications
    bert_config = bert_model.config
    tdo_config = TDOConfig.from_pretrained(f'{MODEL_DIR}/tdo')
    # Text length parameters
    tdo_config.max_length = MAX_LENGTH
    tdo_config.max_neighbors = MAX_NEIGHBORS
    tdo_config.max_position_embeddings = MAX_SENTENCE_LENGTH
    tdo_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    # Transformer parameters
    tdo_config.hidden_size = bert_config.hidden_size
    tdo_config.intermediate_size = bert_config.intermediate_size
    tdo_config.num_attention_heads = bert_config.num_attention_heads
    tdo_config.hidden_act = bert_config.hidden_act
    tdo_config.encoder_layout = ENCODER_LAYOUT
    # Vocabulary parameters
    tdo_config.vocab_size = bert_config.vocab_size
    tdo_config.pad_token_id = bert_config.pad_token_id
    tdo_config.bos_token_id = bert_config.bos_token_id
    tdo_config.eos_token_id = bert_config.eos_token_id
    tdo_config.type_vocab_size = bert_config.type_vocab_size

    # load dummy hi-transformer model
    tdo_model = TDOForMaskedLM.from_config(tdo_config)
    #tdo_model = TDOForSequenceClassification.from_config(tdo_config)

    # copy embeddings
    tdo_model.text_decoder.embeddings.position_embeddings.weight.data[0] = torch.zeros((bert_config.hidden_size,))
    tdo_model.text_decoder.embeddings.position_embeddings.weight.data[1:] = bert_model.bert.embeddings.position_embeddings.weight[1:MAX_LENGTH+tdo_config.pad_token_id+1]
    tdo_model.text_decoder.embeddings.word_embeddings.load_state_dict(bert_model.bert.embeddings.word_embeddings.state_dict())
    tdo_model.text_decoder.embeddings.token_type_embeddings.load_state_dict(bert_model.bert.embeddings.token_type_embeddings.state_dict())
    tdo_model.text_decoder.embeddings.LayerNorm.load_state_dict(bert_model.bert.embeddings.LayerNorm.state_dict())

    # copy transformer layers
    for idx in range(NUM_HIDDEN_LAYERS):
        missing_keys, unexpected_keys = tdo_model.text_decoder.decoder.layer[idx].load_state_dict(bert_model.bert.encoder.layer[idx].state_dict(), strcit=False)
        print(f'{idx}th layer missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}')

    # copy lm_head
    tdo_model.lm_head.dense.load_state_dict(bert_model.cls.predictions.transform.dense.state_dict())
    tdo_model.lm_head.layer_norm.load_state_dict(bert_model.cls.predictions.transform.LayerNorm.state_dict())
    tdo_model.lm_head.decoder.load_state_dict(bert_model.cls.predictions.decoder.state_dict())
    tdo_model.lm_head.bias = copy.deepcopy(bert_model.cls.predictions.bias)

    # save model
    tdo_model.save_pretrained(f'{MODEL_DIR}/PLMs/text-decoder-only-{config.layout}')

    # save tokenizer
    tokenizer.save_pretrained(f'{MODEL_DIR}/PLMs/text-decoder-only-{config.layout}')

    # re-load model
    tdo_model = TDOForMaskedLM.from_pretrained(f'{MODEL_DIR}/PLMs/text-decoder-only-{config.layout}')
    tdo_tokenizer = TDOTokenizer.from_pretrained(f'{MODEL_DIR}/PLMs/text-decoder-only-{config.layout}')
    print(f'TDO model with layout {config.layout} is ready to run!')


if __name__ == '__main__':
    prepare_decoder()
