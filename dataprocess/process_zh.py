import re
from jiayan import load_lm
from jiayan import CharHMMTokenizer

lm = load_lm('jiayan.klm')
tokenizer = CharHMMTokenizer(lm)


def zero_digits(s):
    # return s
    return re.sub('\d', '0', s)


def reverse_sentence(forw_seq):
    backw_seq = []
    for i in range(len(forw_seq) - 1, -1, -1):
        backw_seq.append(forw_seq[i])
    return backw_seq


def reverse_position(forw_position, max_index):
    backw_position = []
    for i in range(len(forw_position)):
        backw_position.append(max_index - forw_position[i])
    return backw_position


def get_seg_features(word_list):
    string = "".join(word_list)
    seg_forw_sentence = []
    seg_forw_positions = []
    word_index = 0
    for word in list(tokenizer.tokenize(string)):
        seg_forw_sentence.append(zero_digits(word.lower()))
        seg_forw_positions.extend([word_index] * len(word))
        word_index += 1

    return seg_forw_sentence, reverse_sentence(seg_forw_sentence), seg_forw_positions, reverse_position(
        seg_forw_positions, word_index - 1)


def word_segment(text):
    word_list = []
    for word in list(tokenizer.tokenize(text)):
        word_list.append(word)
    return word_list


def result_to_json(string, tags):
    item = {"segment": string, "entity_list": []}
    entity_name = ""
    entity_type = ""
    entity_start = 0
    idx = 0
    for tag in tags:
        if tag["tag"][0] == "B":
            if len(entity_name) > 0:
                entity_end = idx - 1
                item["entity_list"].append(
                    {"entity_name": entity_name.strip(), "start_pos": entity_start, "end_pos": entity_end,
                     "type": entity_type})
            entity_name = ""

            entity_name += tag["word"]
            entity_type = tag["tag"][2:]

            entity_start = idx

        elif tag["tag"][0] == "I" and len(entity_name) > 0:

            entity_name += tag["word"]

        else:
            entity_end = idx - 1

            if len(entity_name) > 0:
                item["entity_list"].append(
                    {"entity_name": entity_name.strip(), "start_pos": entity_start, "end_pos": entity_end,
                     "type": entity_type})
            entity_name = ""
            entity_start = idx
        idx += 1
    if len(entity_name) > 0:
        entity_end = idx - 1
        item["entity_list"].append(
            {"entity_name": entity_name.strip(), "start_pos": entity_start, "end_pos": entity_end, "type": entity_type})
    return item
