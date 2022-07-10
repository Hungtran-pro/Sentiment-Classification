from config import *
from lib import *

def normalize(text):
    
	#* Remove các ký tự kéo dài: vd: đẹppppppp
	text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
	
	#* Remove link
	text = ' '.join([w for w in text.split(" ") if len(w) < 8])
    
	#* Chuyển thành chữ thường
	text = text.lower()
	return text

def read_file(path):
    with codecs.open(path, 'r', encoding='UTF-8') as f:
        pos = f.readlines()
        pos_list = [n.strip().replace('\n', '') for n in pos]
    return pos_list

def read_dict(path):
    with open(path, encoding="utf8") as f:
        data = f.read()
        return ast.literal_eval(data)

def write_dict_to_file(path, dict):
    print("Writing a dict to {}".format(path))
    with open(path, "w", encoding="utf8") as file:
        file.write(str(dict))
    print("Done write a dict to {}".format(path))

def write_file(sentences, des_path):

    file = open(des_path, "a", encoding="utf8")

    for sentence in sentences:
        file.write(sentence)

    file.close()

def check_and_create_folder(path):
	if not os.path.exists(path):
		os.mkdir(path)
		print("Create a new {}".format(path))

def check_and_create_file(path):
    if not os.path.exists(path):
        f = open(path, "x")
        print("Create a new {}".format(path))
        f.close()

def split_raw_txt_into_folders(txt_path,folder_path):
    pos_path = folder_path + "\\pos.txt"
    neu_path = folder_path + "\\neu.txt"
    neg_path = folder_path + "\\neg.txt"

    check_and_create_folder(pos_path)
    check_and_create_folder(neu_path)
    check_and_create_folder(neg_path)

    f_txt = open(txt_path, "r", encoding="utf8")
    f_pos = open(pos_path, "a", encoding="utf8")
    f_neu = open(neu_path, "a", encoding="utf8")
    f_neg = open(neg_path, "a", encoding="utf8")

    for sentence in f_txt:
        sentence.lower()
        if "__label__POS":
            f_pos.write(sentence.split("__label__POS")[1].strip())
        if "__label__NEU":
            f_neu.write(sentence.split("__label__NEU")[1].strip())
        if "__label__NEG":
            f_neg.write(sentence.split("__label__NEG")[1].strip())

    f_pos.close()
    f_neu.close()
    f_neg.close()

def get_words_by_vncorenlp(sentences, phase = "word_segment", pos_tag_lst=None, model=None):
    '''
		Input:
			- sentences: a list of string or a string
			- phase: phase for vncorenlp (choose all annotators or just 1)
			- pos_tag_lst: a list of posTag (use to choose words by a specific tag)

		Output:
			- a list of string

		Variable:
			- output_nlp: a return dict by vncorenlp
			- word_nlp: a word seperated by vncorenlp in a sentence
			- output: is a string
			- outputs: is a list of string
	'''
	
	#Only for word segmentation
    if phase == "word_segment":

        #sentences input is a list of string
        if isinstance(sentences, list):
            outputs = []
            for sentence in sentences:
                sentence = normalize(sentence)
                if sentence == "" or sentence == None:
                    continue

                output = model.word_segment(sentence)
                outputs.append(output[0])
            return outputs

        #sentences input is a string
        else:
            sentences = normalize(sentences)
            if sentences == "" or sentences == None:
                return [""]
        return model.word_segment(sentences)
	
	#Get words by POS_TAG
    else:

		#sentences input is a list of string
        if isinstance(sentences, list):
            outputs = []
            for sentence in sentences:
                sentence = normalize(sentence)
                if sentence == "" or sentence == None:
                    continue
                output_nlp = model.annotate_text(sentence)
                output = ""
                for value in output_nlp.values():
                    for word_nlp in value:
                        if all(tag.lower() != word_nlp.get("posTag").lower() for tag in pos_tag_lst):
                            output += (word_nlp.get("wordForm") + " ")
                outputs.append(output)
            return outputs

		#sentences input is a string
        else:
            sentences = normalize(sentences)
            output_nlp = model.annotate_text(sentences)
            output = ""
            for value in output_nlp.values():
                for word_nlp in value:
                    if all(tag.lower() != word_nlp.get("posTag").lower() for tag in pos_tag_lst):
                        output += (word_nlp.get("wordForm") + " ")
        return [output]

def only_segmentation():

    model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)
    for tag in os.listdir(txt_data_path):
        
        check_and_create_file(os.path.join(only_seg_path, tag))
        sentences = read_file(os.path.join(txt_data_path, tag))
        outputs = get_words_by_vncorenlp(sentences=sentences, model=model)
        write_file(sentences=outputs, des_path=os.path.join(only_seg_path, tag))
        print("Write: {}  -  DONE!".format(tag))

#sentences, phase = "word_segment", pos_tag_lst=None, model=None
def pos_tag_without_YCHPE():

    tag_lst = ["Y", "CH", "T", "P", "E"]
    model = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_dir)
    for tag in os.listdir(txt_data_path):

        check_and_create_file(os.path.join(tag_without_YCHTPE_path, tag))
        sentences = read_file(os.path.join(txt_data_path, tag))
        outputs = get_words_by_vncorenlp(sentences=sentences, phase="pos_tag", pos_tag_lst = tag_lst, model=model, )
        write_file(sentences=outputs, des_path=os.path.join(tag_without_YCHTPE_path, tag))
        print("Write: {}  -  DONE!".format(tag))

def get_dict(path, n_gram = 1):
    tag = ["neg.txt", "neu.txt", "pos.txt"]
    des_tag = ["neg_dict.txt", "neu_dict.txt", "pos_dict.txt"]
    des_sum_dict = "sum_dict.txt"
    sum_dict = {}
    for idx in range(0,3):
        dict = {}
        total_words = 0
        total_specific_words = 0
        contents = read_file(os.path.join(path, tag[idx]))
        for content in contents:
            words_lst = content.lower().split()
            for i in range(1, n_gram + 1):
                for j in range(len(words_lst) - i + 1):
                    n_gram_word = ""
                    flag = True
                    for k in range(0, i):
                        if j+k == len(words_lst):
                            flag = False
                            break
                        if k != 0:
                            n_gram_word += " "
                        n_gram_word += words_lst[j+k]
                    if flag:
                        total_words += 1
                        if dict.get(n_gram_word) == None:
                            dict[n_gram_word] = 1
                            total_specific_words += 1
                        else:
                            dict[n_gram_word] += 1
        write_dict_to_file(path=os.path.join(path, des_tag[idx]) ,dict = dict)
        sum_dict[tag[idx].split(".")[0] + "_total_words"] = total_words
        sum_dict[tag[idx].split(".")[0] + "_total_specefic_words"] = total_specific_words
    write_dict_to_file(path=os.path.join(path, des_sum_dict) ,dict = sum_dict)

def get_label_content(sentence):
    if "__label__POS" in sentence:
        return 1, sentence.split("__label__POS")[1].strip()
    if "__label__NEU" in sentence:
        return 0, sentence.split("__label__NEU")[1].strip()
    if "__label__NEG" in sentence:
        return -1, sentence.split("__label__NEG")[1].strip()

'''
POS = 1
NEU = 0
NEG = -1
'''

def n_grams_content(content, n_gram = 1):
    content_lst = content.strip().split(" ")
    result_lst = []
    for i in range(1, n_gram + 1):
        for j in range(len(content_lst) - i + 1):
            n_gram_word = ""
            flag = True
            for k in range(0, i):
                if j+k == len(content_lst):
                    flag = False
                    break
                if k != 0:
                    n_gram_word += " "
                n_gram_word += content_lst[j+k]
            if flag:
                result_lst.append(n_gram_word)
    return result_lst

def get_label(pos, neg):
    if pos >= neg:
        return 1
    else: return -1

# def get_label(pos, neu, neg):
#     if pos > neu and pos > neg: return 1
#     elif neu > neg: return 0
#     else: return -1

def get_predict_label(content_lst):
    pos_dict = read_dict(pos_dict_path)
    neu_dict = read_dict(neu_dict_path)
    neg_dict = read_dict(neg_dict_path)
    sum_dict = read_dict(sum_dict_path)
    possibility_pos = 0
    possibility_neg = 0
    for content in content_lst:
        if content.strip() == "":
            continue
        if pos_dict.get(content) != None:
            possibility_pos += np.log( (pos_dict.get(content) + 1) / (len(content_lst) + sum_dict.get("pos_total_specefic_words")))
        else:
            possibility_pos += np.log(1 / (len(content_lst) + sum_dict.get("pos_total_specefic_words")))
        if neg_dict.get(content) != None:
            possibility_neg += np.log( (neg_dict.get(content) + 1) / (len(content_lst) + sum_dict.get("neg_total_specefic_words")))
        else:
            possibility_neg += np.log(1 / (len(content_lst) + sum_dict.get("neg_total_specefic_words")))
    return get_label(possibility_pos, possibility_neg)

def predict(path, from_file = True, content = None):
    model = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_dir)
    if from_file:
        total_count = 0
        count = 0
        contents = read_file(path)
        for raw_content in contents:
            label, content = get_label_content(raw_content)
            if label == 0:
                continue
            # print(label, get_words_by_vncorenlp(sentences=content, model = model)[0])
            content_nlp = get_words_by_vncorenlp(sentences=content, phase="POS_TAG", pos_tag_lst=["Y", "CH", "T", "P", "E"], model=model)[0]
            content_n_gram = n_grams_content(content=content_nlp, n_gram=3)
            predict_label = get_predict_label(content_lst = content_n_gram)
            # print(content_n_gram)
            print("True label: {} \nPredict label: {}".format(label, predict_label))
            if label == predict_label:
                count += 1
            total_count += 1
            # break
        print("{}/100 accuracy!".format(count * 100 / total_count))
    else:
        # print(label, get_words_by_vncorenlp(sentences=content, model = model)[0])
        content_nlp = get_words_by_vncorenlp(sentences=content, phase="POS_TAG", pos_tag_lst=["Y", "CH", "T", "P", "E"], model=model)[0]
        content_n_gram = n_grams_content(content=content_nlp, n_gram=3)
        predict_label = get_predict_label(content_lst = content_n_gram)
        print(content_n_gram)
        print("Predict label: {}".format(predict_label))

    # nlp_contents = get_words_by_vncorenlp(contents)
    # print(nlp_contents)
        