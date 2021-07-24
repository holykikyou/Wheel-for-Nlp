STOPWORDS=''
def get_stopword_list():
    stop_word_path = STOPWORDS
    stopword_list = [word.replace('\n', '') for word in open(stop_word_path, encoding='utf-8').readlines()]
    return stopword_list


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句a
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]

def re_split(doc,keep_delimiter=False):
    """
    基于正则的简单切分方式
    :param doc:
    :return:
    """
    import re
    if keep_delimiter:
        print("分句保留分隔符")

        sentences = re.split('(。|！|\!|\.|？|\?)', doc)
    else:
        print("分句不保留分隔符")
        sentences= re.split('。|！|\!|\.|？|\?', doc)
    print("分句结果：",sentences)
    return sentences



# 对文本字符级切分，并同词性一并返回 ('苹果','n)
def split_text_to_list(text):
    word_list = psg.cut(text)
    return word_list
def cut_for_search(text):
    import jieba
    word_list=jieba.cut_for_search(text)
    return word_list

# 去除干扰词：过滤除名词外的其他词性，再判断词是否在停用词表中，长度是否大于等于2等。
# 很依赖预先分词的效果即器的性能
def clean_word_list(word_list,noun_only=False):
    stopword_list = get_stopword_list()
    cleaned_word_list = []
    #print('传入对象类型：',type(word_list))
    # 过滤除名词外的其他词性。不进行词性过滤，则将词性都标记为n,表示全部保留
    from jieba.posseg import pair
    for seg in word_list:
        if isinstance(seg,pair):#如果是结巴词性标注分词后的对象
            word = seg.word
            flag = seg.flag
            if (noun_only == True):
                if flag.startswith('n'):
                    # 过滤高停用词表中的词，以及长度为<2的词
                    if word not in stopword_list and len(word) > 1:
                        cleaned_word_list.append(word)
            else:
                if flag.startswith('n') or flag in ['a']:  # 在关键词任务中最好不要使用v词性
                    if word not in stopword_list and len(word) > 1:
                        cleaned_word_list.append(word)
        elif isinstance(seg,str):#如果只是传入词表
            cleaned_word_list.append(seg)



    return cleaned_word_list


def get_word_list_from_doc(text,cut_method='psg'):
    """
    使用不同的分词方式
    从单文本分句构建二维词矩阵
    :return:
    """

    sentence_list = re_split(text)  #分句
    cleaned_word_list=[]
    if cut_method=='psg':
        cleaned_word_list=[clean_word_list(split_text_to_list(sentence)) for sentence in sentence_list]
        return cleaned_word_list    #对每个句子再切分,
    elif cut_method=='cut_for_search':
        cleaned_word_list = [clean_word_list(cut_for_search(sentence)) for sentence in sentence_list]
        return cleaned_word_list

    return cleaned_word_list
