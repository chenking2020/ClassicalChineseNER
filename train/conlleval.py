import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import shutil


def conll_eval(eval_path, result_path):
    eval_perl = os.path.dirname(os.path.abspath(__file__)) + '/conlleval'
    os.system("perl {} < {} > {}".format(eval_perl, eval_path, result_path))

    with open(result_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics


def write_to_ner_result(ner_model, dataset_loader, id_to_tag, tag_result_path):
    ner_model.eval()
    tag_result_file = open(tag_result_path, "w")
    for batch in dataset_loader.iter_batch(shuffle=True):
        strings, feature, decodes, tg = ner_model.eval_batch(batch)
        for sentence, fea, decode, tag in zip(strings, feature, decodes, tg):
            for i in range(len(sentence)):
                if fea[i] == 0:
                    tag_result_file.write("\n")
                    break
                if decode[i] >= len(id_to_tag):
                    tag_result_file.write("{} {} {}\n".format(sentence[i], id_to_tag[tag[i]], "O"))
                else:
                    tag_result_file.write("{} {} {}\n".format(sentence[i], id_to_tag[tag[i]], id_to_tag[decode[i]]))

    tag_result_file.close()


def dev_evaluate(ner_model, dataset_loader, id_to_tag, model_path):
    result_path = os.path.join(model_path, "result")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    tag_result_path = os.path.join(result_path, "tag_result.txt")
    eval_result_path = os.path.join(result_path, "eval_result.txt")
    write_to_ner_result(ner_model, dataset_loader, id_to_tag, tag_result_path)
    metrics_lines = conll_eval(tag_result_path, eval_result_path)
    dev_line = metrics_lines[1]
    f1 = dev_line.split()[-1]
    shutil.rmtree(result_path)
    return float(f1), dev_line
