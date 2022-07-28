from pip import main
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba

def dict_demo():
    data=[{'city': '北京','temperature':100},
          {'city': '上海','temperature':60},
          {'city': '深圳','temperature':30}]
    transfer=DictVectorizer(sparse=False)
    trans_data=transfer.fit_transform(data)
    print("特征名字\n",transfer.get_feature_names_out())
    print(trans_data)
    
    
def english_count_test_demo():
   data = ["life is short,i like like python", "life is too long,i dislike python"]
   transfer=CountVectorizer()
   data=transfer.fit_transform(data)
   print(transfer.get_feature_names_out())
   print("",data.toarray())


def chinese_count_test_demo1():
   data = ["人生苦短 我喜欢python", "正经人谁用python"]
   transfer=CountVectorizer()
   data=transfer.fit_transform(data)
   print(transfer.get_feature_names_out())
   print("",data.toarray())  


def cut_word(sen):
   return " ".join(list(jieba.cut(sen)))


def chinese_count_test_demo2():
   data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
   list=[]
   for temp in data:
      list.append(cut_word(temp))
   transfer=CountVectorizer(stop_words=['一种','还是'])
   data=transfer.fit_transform(list)
   print(transfer.get_feature_names_out())
   print("",data.toarray())  


def tfidf_test_demo():
   data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
   list=[]
   for temp in data:
      list.append(cut_word(temp))
   transfer=TfidfVectorizer()
   data=transfer.fit_transform(list)
   print(transfer.get_feature_names_out())
   print("",data.toarray()) 


if __name__ == '__main__':
    # dict_demo()
    # english_count_test_demo()
    # chinese_count_test_demo()
    # cut_word("傻逼在哪里啊")
   #  chinese_count_test_demo2()
   tfidf_test_demo()