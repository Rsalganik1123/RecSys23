import pickle
from tqdm import tqdm
from utils.scraping_utils import chunks
import argparse
from sentence_transformers import SentenceTransformer

def generate_text_features(data_list, text_key, output_key='text_emb',
                           model_type='distiluse-base-multilingual-cased-v1'):
    """
    :param data_list: list of dictionary with text key
    :param text_key: key name of the text entry to compute the embedding
    :param output_key:  key name to save the embedding
    :param model_type: model type
    :return: updated data list
    """

    # TODO include different models including BOW ...
    model = SentenceTransformer(model_type)
    # ipdb.set_trace() 
    batches = list(chunks(data_list, 32))
    print("start embedding build - {} batches".format(len(batches)))
    for batch in tqdm(batches):
        sentences = [x[text_key] for x in batch]
        sentence_embeddings = model.encode(sentences)
        for data, emb in zip(batch, sentence_embeddings):
            data[output_key] = emb
    return data_list

def generate_text_features_file(input_path, output_path, text_key, output_key):
    """
    :param input_path: list of dictionary with text key
    :param output_path: list of dictionary with text key
    :param text_key: key name of the text entry to compute the embedding
    :param output_key:  key name to save the embedding
    :return:
    """

    print('load data')
    data_list = pickle.load(open(input_path, 'rb')).to_dict('records')

    print('generate embedding')
    data_list = generate_text_features(data_list, text_key, output_key)

    print('Save data')
    pickle.dump(data_list, open(output_path, 'wb'))


