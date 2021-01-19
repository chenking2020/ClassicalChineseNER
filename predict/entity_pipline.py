from __future__ import print_function

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataprocess import data_loader
import json, re
import importlib

id_params_map = {}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
with open(os.path.join(BASE_DIR, "params.json")) as f:
    config_list = json.load(f)
    for config_info in config_list:
        if config_info["active"] == "true":
            id_params_map[config_info["model_id"]] = config_info


class EntityExtractor(object):
    def __init__(self, model_id):
        self.max_word_count = 500
        self.params = id_params_map[model_id]
        self.processor_module = importlib.import_module(
            'ner_provider.dataprocess.process_{}'.format(self.params["lang"]))
        model_path = os.path.join(BASE_DIR, "checkpoint", self.params["model_name"])
        map_path = os.path.join(model_path, "entity_map.json")
        model_file_path = os.path.join(model_path, "entity.model")

        self.params["batch_size"] = 1
        with open(map_path, "r") as f:
            self.entity_map = json.load(f)

        # self.params["batch_size"] = 1

        entity_module = importlib.import_module(
            "ner_provider.model.{}".format(self.params["algo_name"]))
        self.entity_model = entity_module.NERModule(self.params, len(self.entity_map["seg_to_id"]),
                                                    len(self.entity_map["word_to_id"]), self.entity_map["tag_map"])

        self.entity_model.load_checkpoint_file(model_file_path)

    def get_entity_result(self, text):

        sentences = re.split("。|；|！|？", text)
        all_fea_sentences = []
        for sentence in sentences:
            word_list = self.processor_module.word_segment(sentence)
            seg_forw_sentence, seg_backw_sentence, seg_forw_positions, seg_backw_position = self.processor_module.get_seg_features(
                word_list)
            all_fea_sentences.append(
                [sentence, sentence, seg_forw_sentence, seg_backw_sentence, seg_forw_positions, seg_backw_position])
        dataset = data_loader.prepare_dataset(all_fea_sentences, self.entity_map["word_to_id"],
                                              self.entity_map["seg_to_id"], {"O": 0}, self.params["word_pretrain_type"],
                                              self.params["seg_pretrain_type"], False)
        dataset_loader = data_loader.BatchManager(dataset, self.params["batch_size"], False)
        all_result_list = []
        i_sen = 0
        for batch in dataset_loader.iter_batch(shuffle=False):
            strings, feature, decodes, tg = self.entity_model.eval_batch(batch)
            for sentence, fea, decode, tag in zip(strings, feature, decodes, tg):
                result_sentence = []
                for i in range(len(sentence)):
                    if fea[i] == 0:
                        break
                    if decode[i] >= len(self.entity_map["tag_map"]["id_to_tag"]):
                        result_sentence.append({"word": sentence[i], "tag": "O"})
                    else:
                        # 不在预测范围内的tag直接执为O
                        if len(sentence[i].strip()) == 0:
                            result_sentence.append(
                                {"word": sentence[i],
                                 "tag": "O"})
                        else:
                            cur_tag = self.entity_map["tag_map"]["id_to_tag"][str(decode[i])]
                            # if self.predict_entity_types is not None and cur_tag in self.predict_entity_types:
                            result_sentence.append(
                                {"word": sentence[i],
                                 "tag": cur_tag})

                standard_format_out = self.processor_module.result_to_json(sentences[i_sen], result_sentence)
                i_sen += 1
                # if len(self.dic_model.dictionary) == 0:
                #     dic_match_result = final_sentence
                # else:
                #     dic_match_result = self.dic_model.get_ner_result(lang, standard_format_out)

            if len(standard_format_out["entity_list"]) > 0:
                all_result_list.append(standard_format_out)

        return all_result_list
