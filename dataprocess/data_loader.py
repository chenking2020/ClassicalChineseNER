import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import codecs
import math
import random
import re
import importlib


def zero_digits(s):
    # return s
    return re.sub('\\d', '0', s)


def load_sentences(path, lang):
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num += 1
        line = zero_digits(line.rstrip())
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word = line.split()
            if len(word) < 2:
                continue
            assert len(word) >= 2, print(line)
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)

    random.shuffle(sentences)
    return extract_all_features(sentences, lang)


def extract_all_features(sentences, lang):
    processor_module = importlib.import_module('dataprocess.process_{}'.format(lang))
    all_fea_sentences = []
    for sentence in sentences:
        word_list = [w[0] for w in sentence]
        seg_forw_sentence, seg_backw_sentence, seg_forw_positions, seg_backw_position = processor_module.get_seg_features(
            word_list)
        all_fea_sentences.append(
            [sentence, sentence, seg_forw_sentence, seg_backw_sentence, seg_forw_positions, seg_backw_position])
    return all_fea_sentences


def word_mapping(sentences, word_pretrain_type, word_pretrain_path, seg_pretrain_type, seg_pretrain_path):
    if word_pretrain_type is None or len(word_pretrain_type) == 0:
        word_pretrain_type = "pretrain_default"
    word_processor_module = importlib.import_module(
        'dataprocess.process_{}'.format(word_pretrain_type))
    _, word_to_id, id_to_word = word_processor_module.word_mapping(sentences, word_pretrain_path)

    if seg_pretrain_type is None or len(seg_pretrain_type) == 0:
        seg_pretrain_type = "pretrain_default"
    seg_processor_module = importlib.import_module('dataprocess.process_{}'.format(seg_pretrain_type))
    _, seg_to_id, id_to_seg = seg_processor_module.seg_mapping(sentences, seg_pretrain_path)
    return word_to_id, id_to_word, seg_to_id, id_to_seg


def tag_mapping(all_sentences):
    tag_to_id, id_to_tag = {}, {}
    for s in all_sentences:
        for word in s[0]:
            tag = word[-1]
            if tag == "O" and tag not in tag_to_id:
                id = len(tag_to_id)
                tag_to_id[tag] = id
                id_to_tag[id] = tag
    tag_map = {"dico": None, "tag_to_id": tag_to_id, "id_to_tag": id_to_tag}
    return tag_map


def prepare_dataset(sentences, word_to_id, seg_to_id, tag_to_id, word_pretrain_type, seg_pretrain_type, train=True):
    if word_pretrain_type is None or len(word_pretrain_type) > 0:
        word_pretrain_type = "pretrain_default"
    word_processor_module = importlib.import_module(
        'dataprocess.process_{}'.format(word_pretrain_type))
    if seg_pretrain_type is None or len(seg_pretrain_type) > 0:
        seg_pretrain_type = "pretrain_default"
    seg_processor_module = importlib.import_module('dataprocess.process_{}'.format(seg_pretrain_type))

    none_index = tag_to_id["O"]

    data = []
    for i in range(len(sentences)):
        string = []
        for w in sentences[i][1]:
            if isinstance(w, list):
                string.append(w[0])
            else:
                string.append(w)
        words = word_processor_module.sentence_to_word_seq(word_to_id, string)
        forw_segs = seg_processor_module.sentence_to_seg_seq(seg_to_id, sentences[i][2])
        backw_segs = seg_processor_module.sentence_to_seg_seq(seg_to_id, sentences[i][3])
        if train:
            tags = []
            for w in sentences[i][0]:
                if w[-1] in tag_to_id:
                    tags.append(tag_to_id[w[-1]])
                else:
                    tags.append(tag_to_id["O"])
        else:
            tags = [none_index for _ in words]
        data.append([sentences[i][0], string, words, forw_segs, backw_segs, sentences[i][4], sentences[i][5], tags])

    return data


class BatchManager(object):
    def __init__(self, data, batch_size, is_sorted=True):
        self.batch_data = self.sort_and_pad(data, batch_size, is_sorted)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size, is_sorted):
        num_batch = int(math.ceil(len(data) / batch_size))
        if is_sorted:
            sorted_data = sorted(data, key=lambda x: len(x[1]))
        else:
            sorted_data = data
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        sentences = []
        strings = []
        words = []
        forw_segs = []
        backw_segs = []
        forw_positions = []
        backw_positions = []
        targets = []

        max_word_length = max([len(sentence[2]) for sentence in data])
        max_seg_length = max([len(sentence[3]) for sentence in data])
        for line in data:
            sentence, string, word, forw_seg, backw_seg, forw_pos, backw_pos, target = line
            word_padding = [0] * (max_word_length - len(string))
            seg_padding = [0] * (max_seg_length - len(forw_seg))
            sentences.append(sentence)
            strings.append(string + word_padding)
            words.append(word + word_padding)
            forw_segs.append(forw_seg + seg_padding)
            backw_segs.append(backw_seg + seg_padding)
            forw_positions.append(forw_pos + [forw_pos[-1]] * (max_word_length - len(string)))
            backw_positions.append(backw_pos + [backw_pos[-1]] * (max_word_length - len(string)))
            targets.append(target + word_padding)
        return [sentences, strings, words, forw_segs, backw_segs, forw_positions, backw_positions, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
