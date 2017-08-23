#coding=utf-8
import jieba
from gensim.models import Word2Vec
import numpy as np
from os import listdir
from os.path import isfile, join
import requests
import re

class QAExtraction():
    def __init__(self):
        self.qa_map={}
        self.sentences_map={}
        self.sentences=[] #extract
        self.w2v_dim=100
        self.similiarity_value=0.96
        
    def processData(self,corpus_path):
        sentences=[]
        files = [ f for f in listdir(corpus_path) if isfile(join(corpus_path,f)) ]
        for file in files:    
            current_path=join(corpus_path,file)
            print(current_path)
            with open(file=current_path, mode="r", encoding="utf-8") as f:
                for line in f.readlines():
                    tmp_record=line.split(",")
                    sentence=list(jieba.cut(tmp_record[-1].replace("\n","")))
                    sentences.append(sentence)
        
        return sentences
    
    def loadModel(self,model_path):
        self.model = Word2Vec.load(model_path)
        return self.model
    
    def saveModel(self,model_path):
        self.model.save(model_path)
                            
    def train(self,corpus_path="./corpus/"): 
        sentences=self.processData(corpus_path)
        self.model = Word2Vec(sentences, size=self.w2v_dim, window=5, min_count=1, workers=4)
    
    
    def extract(self,file_path):
        sentences_map={}
        with open(file=file_path, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                tmp_record=line.split(",")
                sentence=list(jieba.cut(tmp_record[-1].replace("\n","")))
                sentences_map[tmp_record[0]]=sentence                       #加入弹幕出现的相对时间
    
        self.extractQAPairs(sentences_map,15)       
        
    
    def getSentenceVector(self,sentence):
        sentence_vector=np.array([0.0]*self.w2v_dim)
        for word in sentence:
            sentence_vector+=self.model.wv[word]
            
        return sentence_vector
    def getSentencesSimiliaruiy(self,s1,s2):

        return s1.dot(s2)/(np.sqrt(s1.dot(s1))*np.sqrt(s2.dot(s2)))
        
    def outputQAPairs(self,output_path="./output/output.txt"):
        with open(file=output_path, mode="w", encoding="utf-8") as f:
            for q_sentence in self.qa_map:
                f.write("Q:\t"+q_sentence+"\n")
                for a_sentence in self.qa_map[q_sentence]:
                    f.write("A\t"+"".join(a_sentence)+"\n")

    def isQuestion(self,sentence):
        if(len(sentence)<2):
            return False
        if(sentence[-1]=="吗" or sentence[-1]=="？" or sentence[-1]=='?'): #简单的通过判断 吗 和 ？ 提取问句  
            return True
        else:
            return False
        
    '''
        parm： 
        
        sentences_map：    key：弹幕出现的相对时间  value：对应弹幕的内容
        
        window_size：          弹幕时间窗口大小
    '''
    def extractQAPairs(self,sentences_map,window_size=15):
        
        sentences_sorted=[(k,sentences_map[k]) for k in sorted(sentences_map.keys())]
        question_vector=np.array([0.0]*self.w2v_dim)
        sentence_vector=np.array([0.0]*self.w2v_dim)
        current_question=""
        start_time=0
        
        for sentence in sentences_sorted:
            if self.isQuestion(sentence[1]):
                start_time=float(sentence[0])
                current_question=sentence[1]
                question_vector=self.getSentenceVector(current_question)
            else: 
                appearance_time=float(sentence[0])
                current_sentence=sentence[1]
            
                if appearance_time<(start_time+window_size) and appearance_time>start_time and appearance_time>(start_time+3):
                    #利用word2vec计算句向量
                    sentence_vector=self.getSentenceVector(current_sentence)
                    similiarity_value=self.getSentencesSimiliaruiy(question_vector, sentence_vector)
                    
                    if similiarity_value>self.similiarity_value:
                        try:
                            self.qa_map[''.join(current_question)].append(''.join(current_sentence))
                        except KeyError:
                            self.qa_map[''.join(current_question)]=[]
                            self.qa_map[''.join(current_question)].append(''.join(current_sentence))
                        
                else:
                    continue        
        
        
    
if __name__=="__main__":
    qae=QAExtraction()
#train_path     
#qae.train(train_path)
#qae.saveModel('./output/abc.model')

    qae.loadModel('./output/abc.model')
    qae.extract('D:/Barrage/21883285.txt')
    qae.outputQAPairs('./output/output.txt')
    
