#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')
##part A
from nltk.tokenize import WordPunctTokenizer
def nb_train(x, y):
    ham_count = 0
    spam_count = 0
    ham_fd = nltk.probability.FreqDist()
    spam_fd = nltk.probability.FreqDist()
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    for i in range(len(x)):
        if y[i] == 0:
            ham_count += 1
        elif y[i] == 1:
            spam_count += 1
       
        document = tokenizer.tokenize(x[i].lower())
        for word in document:
            if y[i] == 0:
                ham_fd[word] += 1
            elif y[i] == 1:
                spam_fd[word] += 1

    model = {
        'ham_count': ham_count,
        'spam_count': spam_count,
        'ham_fd': ham_fd,
        'spam_fd': spam_fd
    }

    return model


# In[2]:


import os
def load_data(directory):
    x = []
    y = []
    for file_name in os.listdir(directory):
        with open(os.path.join(directory, file_name), 'r', encoding='latin1') as f:
            email = f.read()
            x.append(email)
            if file_name.startswith("SPAM"):
                y.append(1)
            else:
                y.append(0)
    return x, y


# In[3]:


import math
def calcP(spam_dic,ham_dic,smoothing,use_log):
    
    spam_word_with_p = {}
    ham_word_with_p = {}
    my_S =set(spam_dic | ham_dic)
    total_frequency_spam = sum(spam_dic.values())
    total_frequency_ham = sum(ham_dic.values())
    vocabulary_size = len(my_S)
    
    if use_log:
        for word in my_S:
            spam_word_with_p[word] = math.log10((spam_dic.get(word, 0) + 1) / (total_frequency_spam + vocabulary_size)) if smoothing else math.log10(spam_dic.get(word, 0) / total_frequency_spam) if word in spam_dic else 0
        
        for word in my_S:
            ham_word_with_p[word] = math.log10((ham_dic.get(word, 0) + 1) / (total_frequency_ham + vocabulary_size)) if smoothing else math.log10(ham_dic.get(word, 0) / total_frequency_ham)if word in ham_dic else 0
    
    else:
        for word in my_S:
            spam_word_with_p[word] =((spam_dic.get(word, 0) + 1) / (total_frequency_spam + vocabulary_size)) if smoothing else (spam_dic.get(word, 0) / total_frequency_spam) if word in spam_dic else 0
        
        for word in my_S:
            ham_word_with_p[word] =((ham_dic.get(word, 0) + 1) / (total_frequency_ham + vocabulary_size))if smoothing else (ham_dic.get(word, 0) / total_frequency_ham) if word in ham_dic else 0
            
    
    
    return spam_word_with_p, ham_word_with_p


# In[4]:


import math
##part B
def nb_test1(docs, model, use_log, smoothing):
   
    labels = []
    
    spam_dic = dict(model['spam_fd'].items())
    ham_dic = dict(model['ham_fd'].items())

    spam_dic_w_p,ham_dic_w_p = calcP(spam_dic,ham_dic,smoothing,use_log)
    ##two dictionaries:
    
    #spam_dic_w_p --> It contains the possibility of words that are found in spam, 
    #in addition to the words that are found in ham and are not found in spam
    
    ##ham_dic_w_p -- > It contains the possibility of words that are found in ham, 
    #in addition to the words that are found in  spam and are not found in ham
    
    p_spam = (model['spam_count'] / (model['spam_count'] + model['ham_count']) )
    p_ham = (model['ham_count'] / (model['spam_count'] + model['ham_count']) )
    my_Set =set(spam_dic | ham_dic)
    for i in range(len(docs)):
        spam_prob_log = 0
        ham_prob_log = 0
        spam_prob = 1
        ham_prob = 1
        
        if use_log:
            spam_prob_log += math.log10(p_spam)
            ham_prob_log += math.log10(p_ham)
        else:
            spam_prob *= p_spam
            ham_prob *= p_ham
        
        document = nltk.word_tokenize(docs[i].lower())
        
        for word in document:
            if word in my_Set:
                    if use_log:
                        spam_prob_log += spam_dic_w_p[word] 
                        ham_prob_log += ham_dic_w_p[word] 
                    else:
                        spam_prob *= spam_dic_w_p[word] 
                        ham_prob *= ham_dic_w_p[word] 
        
                    
                             
        
            
        if use_log:
            if spam_prob_log > ham_prob_log:
                labels.append(1)
            else:
                labels.append(0)
        else:
            if spam_prob > ham_prob:
                labels.append(1)
            else:
                labels.append(0)
                   
    return labels


# In[5]:


##part c
def f_score(y_true, y_pred):
    tp = fp = fn = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    f_score = 2 * (precision * recall) / (precision + recall)
    
    return f_score


# In[6]:


#main
def main():
    training_set_path = "D:\Desktop\SPAM_training_set"
    test_set_path ="D:\Desktop\SPAM_test_set"
    x, y = load_data(training_set_path)   
    x1,y1=load_data(test_set_path)
    
    model = nb_train(x, y) 

    y_p_no_log_no_smooth = nb_test1(x1, model, False, False)
    y_p_no_log_smooth = nb_test1(x1, model, False, True)
    y_p_log_no_smooth = nb_test1(x1, model, True, False)
    y_p_log_smooth = nb_test1(x1, model, True, True)

    f1_score_no_log_no_smooth = f_score(y1, y_p_no_log_no_smooth)
    f1_score_no_log_smooth = f_score(y1, y_p_no_log_smooth)
    f1_score_log_no_smooth = f_score(y1, y_p_log_no_smooth)
    f1_score_log_smooth = f_score(y1, y_p_log_smooth)

    #Print the f1-scores
    print("f1-score:")
    print("--> no log _ no smooth:", f1_score_no_log_no_smooth*100)
    print("--> no log _ smooth:", f1_score_no_log_smooth*100)
    print("--> log _ no smooth:", f1_score_log_no_smooth*100)
    print("--> log_smooth:", f1_score_log_smooth*100)


# In[7]:


main()


# In[ ]:




