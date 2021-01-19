from __future__ import print_function

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import importlib
from dataprocess import data_loader
from train.conlleval import dev_evaluate
import datetime
import json


class TrainProcess(object):
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(os.path.join(self.BASE_DIR, "params.json")) as f:
            config_list = json.load(f)
            for config_info in config_list:
                if config_info["active"] == "true":
                    self.params = config_info

    def train_process(self):
        print("{}\tstart reading data...".format(self.get_now_time()))
        model_path = os.path.join(self.BASE_DIR, "checkpoint", self.params["model_name"])
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        all_train_sentences = data_loader.load_sentences(
            os.path.join(self.BASE_DIR, "data", self.params["corpus_name"], "train.txt"), self.params["lang"])
        all_dev_sentences = data_loader.load_sentences(
            os.path.join(self.BASE_DIR, "data", self.params["corpus_name"], "dev.txt"), self.params["lang"])

        word_to_id, id_to_word, seg_to_id, id_to_seg = data_loader.word_mapping(all_train_sentences,
                                                                                self.params["word_pretrain_type"],
                                                                                os.path.join(
                                                                                    self.params["pretrain_basedir"],
                                                                                    self.params[
                                                                                        "word_pretrain_name"]),
                                                                                self.params["seg_pretrain_type"],
                                                                                os.path.join(
                                                                                    self.params["pretrain_basedir"],
                                                                                    self.params["seg_pretrain_name"]))

        source_tag_map = data_loader.tag_mapping(all_train_sentences)

        train_data = data_loader.prepare_dataset(
            all_train_sentences, word_to_id, seg_to_id,
            source_tag_map["tag_to_id"], self.params["word_pretrain_type"], self.params["seg_pretrain_type"]
        )
        dev_data = data_loader.prepare_dataset(
            all_dev_sentences, word_to_id, seg_to_id,
            source_tag_map["tag_to_id"], self.params["word_pretrain_type"], self.params["seg_pretrain_type"]
        )

        train_manager = data_loader.BatchManager(train_data, self.params["batch_size"])
        dev_manager = data_loader.BatchManager(dev_data, self.params["batch_size"])

        print("{}\tfinish reading data!, percent of train/dev: {}/{}".format(self.get_now_time(),
                                                                             len(all_train_sentences),
                                                                             len(all_dev_sentences)))
        # ToDo 这里核心是根据task_id读取到所用到的模型，灵活初始化模型
        print("{}\tstart init model...".format(self.get_now_time()))
        ner_module = importlib.import_module(
            "model.{}".format(self.params["algo_name"]))
        ner_model = ner_module.NERModule(self.params, len(seg_to_id), len(word_to_id), source_tag_map)

        ner_model.init_word_embedding(self.params["word_pretrain_type"],
                                      os.path.join(self.params["pretrain_basedir"], self.params["word_pretrain_name"]))
        ner_model.init_seg_embedding(self.params["seg_pretrain_type"],
                                     os.path.join(self.params["pretrain_basedir"], self.params["seg_pretrain_name"]))

        ner_model.set_optimizer()

        print("{}\tfinished init model!".format(self.get_now_time()))

        tot_length = len(all_train_sentences)

        best_f1 = float('-inf')
        best_f1_epoch = 0
        patience_count = 0
        print("{}\tstart training...".format(self.get_now_time()))
        for epoch_idx in range(self.params["epoch"]):
            epoch_loss = 0
            ner_model.start_train_setting()
            iter_step = 0
            for batch in train_manager.iter_batch(shuffle=True):
                iter_step += 1
                loss = ner_model.train_batch(batch)
                epoch_loss += loss
                print("{}\t{}".format(self.get_now_time(),
                                      "epoch: %s, current step: %s, current loss: %.4f" % (
                                          epoch_idx, iter_step, loss / len(batch))))
                # print("epoch: %s, current step: %s, current loss: %.4f time use: %s" % (
                #     epoch_idx, iter_step, loss / len(batch),
                #     time.time() - step_start_time))

            epoch_loss /= tot_length

            update_lr = ner_model.end_train_setting()

            # eval & save check_point'
            dev_f1, dev_line = dev_evaluate(ner_model, dev_manager,
                                            source_tag_map["id_to_tag"], model_path)

            print("{}\t{}".format(self.get_now_time(), dev_line))
            # print("dev: {}".format(dev_line))

            if dev_f1 > best_f1:
                best_f1 = dev_f1
                best_f1_epoch = epoch_idx
                patience_count = 0
                print("{}\t{}".format(self.get_now_time(), 'best average f1: %.4f in epoch_idx: %d , saving...' % (
                    best_f1, best_f1_epoch)))
                # print('best average f1: %.4f in epoch_idx: %d , saving...' % (best_f1, best_f1_epoch))

                try:
                    ner_model.save_checkpoint({
                        'word_to_id': word_to_id,
                        'id_to_word': id_to_word,
                        'seg_to_id': seg_to_id,
                        'id_to_seg': id_to_seg,
                        'tag_map': source_tag_map
                    }, os.path.join(model_path, "entity"))
                except Exception as inst:
                    print(inst)
            else:
                patience_count += 1
                print("{}\t{}".format(self.get_now_time(),
                                      'poor current average f1: %.4f, best average f1: %.4f in epoch_idx: %d' % (
                                          dev_f1, best_f1, best_f1_epoch)))

                # print(
                #     'poor current average f1: %.4f, best average f1: %.4f in epoch_idx: %d' % (
                #         dev_f1, best_f1, best_f1_epoch))

            # print('epoch: ' + str(epoch_idx) + '\t in ' + params["epoch"] + ' take: ' + str(
            #     time.time() - start_time) + ' s')

            if patience_count >= self.params["patience"] and epoch_idx >= self.params["least_iters"]:
                print("{}\tfinished training！！！".format(self.get_now_time()))
                break

    def get_now_time(self):
        timenow = (datetime.datetime.utcnow() + datetime.timedelta(hours=8))
        now_time_str = timenow.strftime("%Y-%m-%d %H:%M:%S")
        return now_time_str


if __name__ == '__main__':
    train_i = TrainProcess()
    train_i.train_process()
