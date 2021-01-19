import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.autograd as autograd
import json


class NERModule(nn.Module):
    def __init__(self, params, seg_size, vocab_size, source_tag_map):

        super(NERModule, self).__init__()
        self.params = params
        self.seg_embeds = nn.Embedding(seg_size, self.params["seg_dim"])
        self.forw_seg_lstm = nn.LSTM(input_size=self.params["seg_dim"], hidden_size=self.params["seg_hidden"] // 2,
                                     num_layers=self.params["seg_layers"],
                                     bidirectional=False, batch_first=True)
        self.backw_seg_lstm = nn.LSTM(input_size=self.params["seg_dim"], hidden_size=self.params["seg_hidden"] // 2,
                                      num_layers=self.params["seg_layers"],
                                      bidirectional=False, batch_first=True)

        self.word_embeds = nn.Embedding(vocab_size, self.params["word_dim"])

        self.word_lstm = nn.LSTM(self.params["word_dim"] + self.params["seg_hidden"],
                                 self.params["word_hidden"] // 2,
                                 num_layers=self.params["word_layers"],
                                 bidirectional=True, batch_first=True)

        self.word_rnn_layers = self.params["word_layers"]

        self.dropout = nn.Dropout(p=self.params["drop_out"])

        if self.params["gpu"] >= 0:
            torch.cuda.set_device(self.params["gpu"])
            self.seg_embeds = self.seg_embeds.cuda()
            self.forw_seg_lstm = self.forw_seg_lstm.cuda()
            self.backw_seg_lstm = self.backw_seg_lstm.cuda()
            self.word_embeds = self.word_embeds.cuda()
            self.word_lstm = self.word_lstm.cuda()
            self.dropout = self.dropout.cuda()

        tag_map = source_tag_map["tag_to_id"]
        if "START" not in tag_map:
            tag_map["START"] = len(tag_map)
        if "STOP" not in tag_map:
            tag_map["STOP"] = len(tag_map)
        self.crf_model = CRF(self.params["word_hidden"], tag_map)
        self.batch_size = 1
        self.word_seq_length = 1

    def set_batch_seq_size(self, sentence):
        tmp = sentence.size()
        self.batch_size = tmp[0]
        self.word_seq_length = tmp[1]

    def init_word_embedding(self, pretrain_type, pretrain_path):
        if pretrain_type == "word_vector":
            with open(pretrain_path, "r") as f:
                vector_info = f.readline()
                vector_info = vector_info.strip().split()
                weights = torch.zeros(size=[int(vector_info[0]) + 2, int(vector_info[1])], dtype=torch.float)
                weights[0] = torch.rand(size=[int(vector_info[1])])
                weights[1] = torch.rand(size=[int(vector_info[1])])
                idx = 2
                for line in f:
                    line = line.strip().split()
                    weights[idx] = torch.tensor([float(v) for v in line[1:]])
                    idx += 1

            if self.params["gpu"] >= 0:
                self.word_embeds.weight = nn.Parameter(weights.cuda())
            else:
                self.word_embeds.weight = nn.Parameter(weights)

        else:
            nn.init.uniform_(self.word_embeds.weight, -0.25, 0.25)

    def init_seg_embedding(self, pretrain_type, pretrain_path):
        if pretrain_type == "word_vector":
            with open(pretrain_path, "r") as f:
                vector_info = f.readline()
                vector_info = vector_info.strip().split()
                weights = torch.zeros(size=[int(vector_info[0]) + 2, int(vector_info[1])], dtype=torch.float)
                weights[0] = torch.rand(size=[int(vector_info[1])])
                weights[1] = torch.rand(size=[int(vector_info[1])])
                idx = 2
                for line in f:
                    line = line.strip().split()
                    weights[idx] = torch.tensor([float(v) for v in line[1:]])
                    idx += 1

            if self.params["gpu"] >= 0:
                self.seg_embeds.weight = nn.Parameter(weights.cuda())
            else:
                self.seg_embeds.weight = nn.Parameter(weights)

        else:
            nn.init.uniform_(self.seg_embeds.weight, -0.25, 0.25)

    def set_optimizer(self):
        if self.params["update"] == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.params["lr"], momentum=self.params["momentum"])
        elif self.params["update"] == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr"])

    def start_train_setting(self):
        self.train()

    def train_batch(self, batch):
        _, _, w_f, s_f, s_b, f_p, b_p, tg_v = batch
        self.zero_grad()
        logits = self(w_f, s_f, s_b, f_p, b_p)
        loss = self.crf_model.get_crf_loss(logits, tg_v)
        epoch_loss = self.to_scalar(loss)
        loss.backward()
        self.clip_grad_norm()
        self.optimizer.step()
        return epoch_loss

    def end_train_setting(self):
        new_learning_rate = self.params["lr"]
        torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        for param_group in self.optimizer.param_groups:
            new_learning_rate = param_group['lr']
            break
        return new_learning_rate

    def eval_batch(self, batch):
        _, strings, feature, seg_forw_feature, seg_backw_feature, seg_forw_position, seg_backw_position, tg = batch
        logits = self(feature, seg_forw_feature, seg_backw_feature, seg_forw_position, seg_backw_position)
        decodes = []
        for sentence, fea, logit, tag in zip(strings, feature, logits, tg):
            _, decode = self.crf_model.viterbi_decode(logit)
            decodes.append(decode)
        return strings, feature, decodes, tg

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.params["clip_grad"])

    def to_scalar(self, var):
        return var.view(-1).data.tolist()[0]

    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, data_map, filename):
        with open(filename + '_map.json', 'w') as f:
            f.write(json.dumps(data_map, ensure_ascii=False))
        torch.save(self.state_dict(), filename + '.model')

    def load_checkpoint_file(self, model_path):
        checkpoint_file = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint_file)
        self.eval()

    def forward(self, word_seq, forw_sentence, back_sentence, forw_position, back_position, hidden=None):

        word_seq = torch.LongTensor(word_seq)
        forw_sentence = torch.LongTensor(forw_sentence)
        back_sentence = torch.LongTensor(back_sentence)
        forw_position = torch.LongTensor(forw_position)
        back_position = torch.LongTensor(back_position)
        if self.params["gpu"] >= 0:
            word_seq = autograd.Variable(word_seq).cuda()
            forw_sentence = autograd.Variable(forw_sentence).cuda()
            back_sentence = autograd.Variable(back_sentence).cuda()
            forw_position = autograd.Variable(forw_position).cuda()
            back_position = autograd.Variable(back_position).cuda()
        else:
            word_seq = autograd.Variable(word_seq)
            forw_sentence = autograd.Variable(forw_sentence)
            back_sentence = autograd.Variable(back_sentence)
            forw_position = autograd.Variable(forw_position)
            back_position = autograd.Variable(back_position)

        self.set_batch_seq_size(forw_position)

        forw_emb = self.seg_embeds(forw_sentence)
        back_emb = self.seg_embeds(back_sentence)

        # # dropout
        # d_f_emb = self.dropout(forw_emb)
        # d_b_emb = self.dropout(back_emb)

        forw_lstm_out, _ = self.forw_seg_lstm(forw_emb)

        back_lstm_out, _ = self.backw_seg_lstm(back_emb)

        # select predict point
        forw_position = forw_position.unsqueeze(2).expand(self.batch_size, self.word_seq_length,
                                                          self.params["seg_hidden"] // 2)
        select_forw_lstm_out = torch.gather(forw_lstm_out, 1, forw_position)

        back_position = back_position.unsqueeze(2).expand(self.batch_size, self.word_seq_length,
                                                          self.params["seg_hidden"] // 2)
        select_back_lstm_out = torch.gather(back_lstm_out, 1, back_position)

        # fb_lstm_out = self.dropout(torch.cat((select_forw_lstm_out, select_back_lstm_out), dim=2))
        fb_lstm_out = torch.cat((select_forw_lstm_out, select_back_lstm_out), dim=2)

        # word
        word_emb = self.word_embeds(word_seq)
        # d_word_emb = self.dropout(word_emb)

        # combine
        word_input = torch.cat((word_emb, fb_lstm_out), dim=2)

        # word level lstm
        lstm_out, _ = self.word_lstm(self.dropout(word_input))
        # d_lstm_out = self.dropout(lstm_out)

        # convert to crf
        logits = self.crf_model(lstm_out.cpu())

        return logits


class CRF(nn.Module):
    def __init__(self, hidden_dim, tag_map):
        super(CRF, self).__init__()
        self.tag_map = tag_map
        self.tagset_size = len(self.tag_map)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )
        self.transitions.data[self.tag_map["START"], :] = -1000.
        self.transitions.data[:, self.tag_map["STOP"]] = -1000.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=True)

    def forward(self, lstm_out):
        logits = self.hidden2tag(lstm_out.cpu())
        return logits

    def log_sum_exp(self, vec):
        max_score = torch.max(vec, 0)[0].unsqueeze(0)
        max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
        result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
        return result.squeeze(1)

    def real_path_score(self, logits, label):

        score = torch.zeros(1)
        label = torch.cat([torch.tensor([self.tag_map["START"]], dtype=torch.long), label])
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score
        score += self.transitions[label[-1], self.tag_map["STOP"]]
        return score

    def total_score(self, logits):

        previous = torch.full((1, self.tagset_size), 0)
        for index in range(len(logits)):
            previous = previous.expand(self.tagset_size, self.tagset_size).t()
            obs = logits[index].view(1, -1).expand(self.tagset_size, self.tagset_size)
            scores = previous + obs + self.transitions
            previous = self.log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag_map["STOP"]]
        # caculate total_scores
        total_scores = self.log_sum_exp(previous.t())[0]
        return total_scores

    def get_crf_loss(self, logits, tags):

        tags = torch.LongTensor(tags)
        real_path_score = torch.zeros(1)
        total_score = torch.zeros(1)
        for logit, tag in zip(logits, tags):
            real_path_score += self.real_path_score(logit, tag)
            total_score += self.total_score(logit)
        return total_score - real_path_score

    def viterbi_decode(self, logits):

        trellis = torch.zeros(logits.size())

        backpointers = torch.zeros(logits.size(), dtype=torch.long)

        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi
