import argparse

import torch
import copy
import warnings
from transformers import EncoderDecoderModel, AutoModelForMaskedLM, AutoTokenizer
from model import TDOForMaskedLM, TDOConfig

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
    parser.add_argument('--description', default='default')
    parser.add_argument('--random_init', default='True')
    parser.add_argument('--layout', default='s1', choices=['s1', 's2', 'p1', 'p2', 'e1', 'e2',
                                                           'l1', 'l2', 'b1', 'b2', 'f12', 'f8', 'f6'],
                        help='S|D encoders layout')
    parser.add_argument('--max_length', default=512)
    parser.add_argument('--max_seq_length', default=64)
    parser.add_argument('--max_neighbors', default=64)
    parser.add_argument('--lora_type', default='none')
    config = parser.parse_args()

    MAX_LENGTH = int(config.max_length)
    MAX_SEQ_LENGTH = int(config.max_seq_length)
    MAX_NEIGHBORS = int(config.max_neighbors)
    LORA_TYPE = str(config.lora_type)
    ENCODER_LAYOUT = {}
    for idx, block_pattern in enumerate(LAYOUTS[config.layout].split('|')):
        ENCODER_LAYOUT[str(idx)] = {"attention": True if 'S' in block_pattern else False,
                                    "crossattention": True if 'D' in block_pattern else False}
    NUM_HIDDEN_LAYERS = len(ENCODER_LAYOUT.keys())

    # load pre-trained bert model and tokenizer
    if config.random_init == 'True':
        BERT_LAYERS = NUM_HIDDEN_LAYERS + 1 if NUM_HIDDEN_LAYERS % 2 else NUM_HIDDEN_LAYERS
        BERT_CHECKPOINT = f'google/bert_uncased_L-{str(BERT_LAYERS)}_H-768_A-12'
        tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT, model_max_length=MAX_LENGTH)
        bert_model = AutoModelForMaskedLM.from_pretrained(BERT_CHECKPOINT)
    else:
        BERT_CHECKPOINT = f'patrickvonplaten/bert2bert_cnn_daily_mail'
        tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT, model_max_length=MAX_LENGTH)
        bert_model = EncoderDecoderModel.from_pretrained(BERT_CHECKPOINT).decoder

    # load dummy config and change specifications
    bert_config = bert_model.config
    tdo_config = TDOConfig.from_pretrained(f'{MODEL_DIR}/tdo')
    # Text length parameters
    tdo_config.max_length = MAX_LENGTH
    tdo_config.max_seq_length = MAX_SEQ_LENGTH
    tdo_config.max_position_embeddings = MAX_LENGTH
    tdo_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    # Neighbor parameters
    tdo_config.neighbor_max = MAX_NEIGHBORS
    tdo_config.neighbor_hidden_size = bert_config.hidden_size
    # LORA parameters
    tdo_config.lora_type = LORA_TYPE
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
    tdo_model.text_decoder.embeddings.position_embeddings.weight.data = bert_model.bert.embeddings.position_embeddings.weight[:MAX_LENGTH]
    tdo_model.text_decoder.embeddings.word_embeddings.load_state_dict(bert_model.bert.embeddings.word_embeddings.state_dict())
    tdo_model.text_decoder.embeddings.token_type_embeddings.load_state_dict(bert_model.bert.embeddings.token_type_embeddings.state_dict())
    tdo_model.text_decoder.embeddings.LayerNorm.load_state_dict(bert_model.bert.embeddings.LayerNorm.state_dict())

    # copy transformer layers
    for idx in range(NUM_HIDDEN_LAYERS):
        missing_keys, unexpected_keys = tdo_model.text_decoder.decoder.layer[idx].load_state_dict(bert_model.bert.encoder.layer[idx].state_dict(), strict=False)
        print(f'{idx}th layer missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}')

    # copy lm_head
    tdo_model.lm_head.dense.load_state_dict(bert_model.cls.predictions.transform.dense.state_dict())
    tdo_model.lm_head.layer_norm.load_state_dict(bert_model.cls.predictions.transform.LayerNorm.state_dict())
    tdo_model.lm_head.decoder.load_state_dict(bert_model.cls.predictions.decoder.state_dict())
    tdo_model.lm_head.bias = copy.deepcopy(bert_model.cls.predictions.bias)

    # save model
    tdo_model.save_pretrained(f'{MODEL_DIR}/PLMs/text-decoder-only-{config.layout}-{config.description}')

    # save tokenizer
    tokenizer.save_pretrained(f'{MODEL_DIR}/PLMs/text-decoder-only-{config.layout}-{config.description}')

    # re-load model
    tdo_model = TDOForMaskedLM.from_pretrained(f'{MODEL_DIR}/PLMs/text-decoder-only-{config.layout}-{config.description}')
    tdo_tokenizer = AutoTokenizer.from_pretrained(f'{MODEL_DIR}/PLMs/text-decoder-only-{config.layout}-{config.description}')
    print(f'TDO model with layout {config.layout} is ready to run!')


if __name__ == '__main__':
    prepare_decoder()
