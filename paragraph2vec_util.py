#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

from gensim.models import doc2vec
import logging
import sys
import nltk
import re
from word2vec_util import Word2VecUtil

class Paragraph2VecUtil(Word2VecUtil):

    PUNKT_TOKENIZER_LOCATION = 'tokenizers/punkt/english.pickle'

    def __init__(self):
        Word2VecUtil.__init__(self)

    def create_sentence_data(self, corpus_file, save_path):
        self.logger.info("reading corpus file")
        uid = 0
        sent_detector = nltk.data.load(Paragraph2VecUtil.PUNKT_TOKENIZER_LOCATION)
        self.logger.info("creating sentence data")
        pat = re.compile("^$")
        f = open(save_path, 'w')
        for line_id, line in enumerate(open(corpus_file)):
            if pat.match(line):
                continue
            sentences = sent_detector.tokenize(line.strip().decode('utf-8'))
            for sent in sentences:
                self.logger.info("line_id: %10d, uid: %10d", line_id, uid)
                f.write(str(uid) + "\t" + sent.encode('utf-8') + "\n")
                if sys.maxint -1 == uid:
                    raise Exception('uid exceeded maxint')
                uid += 1
        f.close()
        self.logger.info("created sentence data")

    def __sentence_iter(self, sentence_file):
        self.logger.info("reading sentence file")
        for line in open(sentence_file):
            uid = line.split()[0]
            sent = line.split()[1:]
            yield doc2vec.LabeledSentence(words=sent, labels=['SENT_%s' % uid])

    def learn_paragrah2vec(self, corpus_file, save_file=None):
        self.logger.info("reading corpus file")
        sentences = self.__sentence_iter(corpus_file)
        self.logger.info("learning paragraph2vec")
        self.model = doc2vec.Doc2Vec(sentences, alpha=0.025, min_alpha=0.025)
        self.logger.info("saving model data")
        if save_file:
            self.model.save(save_file)
        self.logger.info("finished word2vec learning")


    def load_model(self, save_file):
        pass


if __name__=="__main__":

    logging.root.setLevel(level=logging.INFO)

    p2v = Paragraph2VecUtil()
    p2v.learn_paragrah2vec("word2vec_models/sentence_data.txt", "word2vec_models/enwiki_paragraph2vec")

