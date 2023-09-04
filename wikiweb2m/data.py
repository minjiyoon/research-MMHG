import time
import torch
from transformers import AutoTokenizer
import pickle
import pandas as pd
from PIL import Image
from urllib.request import urlopen

from language_modelling import utils

def load_wikiweb2m_tf(task):
    with open(f'./wikiweb2m/raw/wikiweb2m_{task}_train_medium.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(f'./wikiweb2m/raw/wikiweb2m_{task}_val_medium.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    with open(f'./wikiweb2m/raw/wikiweb2m_{task}_test_medium.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    return train_dataset, val_dataset[:10000], test_dataset[:10000]

def load_wikiweb2m(task):
    train_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_train_large.parquet')
    val_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_val_large.parquet')
    test_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_test_large.parquet')

    with open(f'./wikiweb2m/raw/{task}_id_split_large.pkl', 'rb') as f:
        id_list = pickle.load(f)

    return train_df, val_df, test_df, id_list


class WikiWeb2M(torch.utils.data.Dataset):

    def __init__(self, args, df, id_list, tokenizer, feature_extractor_model=None, decoder_only=False):
        self.path = './wikiweb2m/raw/'
        self.task = args.task
        self.context = args.context
        self.decoder_only = decoder_only

        self.df = df
        self.id_list = id_list
        self.tokenizer = tokenizer
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length

        if feature_extractor_model is not None and self.context in ('section_all', 'all'):
            self.feature_extractor = utils.get_feature_extractor_for_model(feature_extractor_model)
            self.n_visual_tokens = args.n_visual_tokens
            self.max_images = 5

    def __len__(self):
        return len(self.id_list)

    def get_page_info(self, d):
        page_url = d['page_url'].decode()
        page_title = d['page_title'].decode()
        page_description = d['page_description'].decode()
        #return ", ".join([page_title, page_url, page_description])
        return ", ".join([page_title, page_description])

    def get_section_info(self, section_id, d, remove_summary=True):
        page_title = d['page_title'].decode()
        section_title = d['section_title'][section_id].decode()
        section_depth = str(d['section_depth'][section_id])
        section_heading = str(d['section_heading'][section_id])
        section_parent_index = str(d['section_parent_index'][section_id])
        section_summary = d['section_summary'][section_id].decode()
        section_rest_sentence = d['section_rest_sentence'][section_id].decode()
        if remove_summary:
            section_info = ", ".join([section_rest_sentence])
            section_info, section_summary = ' '.join(section_info.replace('\n', '').split()), ' '.join(section_summary.replace('\n', '').split())
            return section_info, section_summary
        else:
            section_info = ", ".join([section_summary, section_rest_sentence])
            section_info = ' '.join(section_info.replace('\n', '').split())
            return section_info

    def get_section_images(self, page_id, section_id, d):
        section_images = []
        section_captions = []
        section_num = d['section_title'].shape[0]
        image_urls = d['image_url'].reshape(section_num, -1)
        image_captions = d['image_caption'].reshape(section_num, -1)
        for image_id in range(image_urls[section_id].shape[0]):
            if image_urls[section_id][image_id] == b'':
                continue
            image_caption = image_captions[section_id][image_id].decode()
            section_captions.append(image_caption)

            image_url = image_urls[section_id][image_id].decode()
            try:
                img = Image.open(urlopen(image_url))
            except:
                time.sleep(4)
                img = Image.open(urlopen(image_url))
            #file_format = os.path.splitext(image_url)[1][1:]
            #img = Image.open('./wikiweb2m/raw/images/{page_id}_{section_id}_{image_id}.{file_format}')
            image = utils.get_pixel_values_for_model(self.feature_extractor, img)
            section_images.append(image)
            # one image per section
            break
        return ", ".join(section_captions), section_images

    def get_image_info(self, section_id, image_id, d, remove_caption=True):
        image_url =  tf.sparse.to_dense(d[1]['section_image_url'])[section_id][image_id].numpy().decode()
        image_caption = tf.sparse.to_dense(d[1]['section_image_captions'])[section_id][image_id].numpy().decode()
        if remove_caption:
            return Image.open(urlopen(image_url))
        else:
            return Image.open(urlopen(image_url)), image_caption

    def __getitem__(self, index):
        if self.task == "section":
            page_id, section_id = self.id_list[index]
            d = self.df[self.df['page_id'] == page_id].iloc[0]
            images = None
            if self.context == "section_only":
                section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
                inputs = "summarize: " + section_info
                inputs = ' '.join(inputs.replace('\n', '').split())
                input_ids = self.tokenizer(inputs, max_length=self.max_input_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]

            elif self.context == "section_all":
                section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
                image_captions, images = self.get_section_images(page_id, section_id, d)
                inputs = "summarize: " + section_info + image_captions

                max_text_len = self.max_input_length - len(images) * self.n_visual_tokens
                input_ids = self.tokenizer(inputs, max_length=max_text_len, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
                image_range = torch.LongTensor([[input_ids.shape[0], input_ids.shape[0]]])
                if len(images) > 0 :
                    input_ids = torch.cat([input_ids, torch.LongTensor(len(images) * self.n_visual_tokens * [-1])], dim=0)
                    image_range[0][1] = input_ids.shape[0]
                else:
                    # wikiweb2m image padding size
                    images = [torch.zeros((3,  224, 224))]

            elif self.context == "text_only":
                page_info = self.get_page_info(d)
                section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
                context_info = []
                for context_id in range(len(d['section_title'])):
                    if context_id == section_id:
                        continue
                    context_info.append(self.get_section_info(context_id, d, remove_summary=False))
                context_info = ', '.join(context_info)
                inputs = "summarize: " + section_info + ", context: " + page_info + context_info
                inputs = ' '.join(inputs.replace('\n', '').split())
                input_ids = self.tokenizer(inputs, max_length=self.max_input_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]

            elif self.context == "all":
                page_info = self.get_page_info(d)
                section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
                image_captions, images = self.get_section_images(page_id, section_id, d)
                inputs = "summarize: " + section_info + image_captions

                input_ids = self.tokenizer(inputs, max_length=self.max_input_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
                image_range = [[input_ids.shape[0], input_ids.shape[0]]]
                if len(images) > 0:
                    input_ids = torch.cat([input_ids, torch.LongTensor(len(images) * self.n_visual_tokens * [-1])], dim=0)
                    image_range[0][1] = input_ids.shape[0]
                else:
                    # wikiweb2m image padding size
                    images = [torch.zeros((3 * 224 * 224))]

                for context_id in range(len(d['section_title'])):
                    if context_id == section_id:
                        continue
                    context_info = self.get_section_info(context_id, d, remove_summary=False)
                    image_captions, new_images = self.get_section_images(page_id, context_id, d)
                    if len(images) == 1:
                        inputs = "context: " + context_info + image_captions
                    else:
                        inputs = ", " + context_info + image_captions

                    new_input_ids = self.tokenizer(inputs, max_length=self.max_input_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
                    input_ids = torch.cat([input_ids, new_input_ids], dim=0)
                    image_range.append([input_ids.shape[0], input_ids.shape[0]])
                    if len(new_images) > 0 :
                        input_ids = torch.cat([input_ids, torch.LongTensor(len(new_images) * self.n_visual_tokens * [-1])], dim=0)
                        image_range[-1][1] = input_ids.shape[0]
                        images.extend(new_images)
                    else:
                        # wikiweb2m image padding size
                        images.extend([torch.zeros((3 * 224 * 224))])
                if len(input_ids) > self.max_input_length:
                    input_ids = input_ids[:self.max_input_length]

        # OPT
        if self.decoder_only:
            labels = "summary: " + labels
            labels = ' '.join(labels.replace('\n', '').split())
            label_ids = self.tokenizer(labels, max_length=self.max_output_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
            sep_id = input_ids.shape[0]
            input_ids = torch.cat([input_ids, label_ids[1:], torch.LongTensor([self.tokenizer.eos_token_id])], dim=0)
            model_inputs = self.tokenizer.pad({"input_ids": [input_ids]}, max_length=self.max_input_length + self.max_output_length, padding="max_length", return_tensors="pt")
            return {"input_ids": model_inputs.input_ids[0], "attention_mask": model_inputs.attention_mask[0], "labels": model_inputs.input_ids[0],"sep_id": sep_id}

        # Padding
        model_inputs = self.tokenizer.pad({"input_ids": [input_ids]}, max_length=self.max_input_length, padding="max_length", return_tensors="pt")

        labels = ' '.join(labels.replace('\n', '').split())
        label_ids = self.tokenizer(labels, max_length=self.max_output_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
        labels = self.tokenizer.pad({"input_ids": [label_ids]}, max_length=self.max_output_length, padding="max_length", return_tensors="pt").input_ids[0]
        labels_with_ignore_index = [label if label != 0 else -100 for label in labels]

        input_ids = model_inputs.input_ids[0]
        attention_mask = model_inputs.attention_mask[0]
        labels = torch.LongTensor(labels_with_ignore_index)

        if self.context in ("section_all", "all"):
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "images": images, "image_ranges": image_range}
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


