def create_dico(item_list):
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def word_mapping(sentences, pretrain_path):
    words = [[x[0].lower() for x in s[0]] for s in sentences]
    word_dico = create_dico(words)
    word_dico["<PAD>"] = 100000001
    word_dico['<UNK>'] = 100000000
    word_to_id, id_to_word = create_mapping(word_dico)

    return word_dico, word_to_id, id_to_word


def seg_mapping(sentences, pretrain_path):
    segs = [[x.lower() for x in s[2]] for s in sentences]
    seg_dico = create_dico(segs)
    seg_dico["<PAD>"] = 100000001
    seg_dico['<UNK>'] = 100000000
    seg_to_id, id_to_seg = create_mapping(seg_dico)
    return seg_dico, seg_to_id, id_to_seg


def sentence_to_word_seq(word_to_id, word_list):
    return [word_to_id[w.lower() if w.lower() in word_to_id else '<UNK>']
            for w in word_list]


def sentence_to_seg_seq(seg_to_id, seg_list):
    return [seg_to_id[w.lower() if w.lower() in seg_to_id else '<UNK>']
            for w in seg_list]
