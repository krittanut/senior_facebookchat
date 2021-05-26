#Python libraries that we need to import for our bot
import random
from flask import Flask, request
from pymessenger.bot import Bot
import os 
import fasttext
import re
import emoji
import pickle
import numpy as np
from pythainlp.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
import joblib
import sklearn 
app = Flask(__name__)
ACCESS_TOKEN = 'EAATdr7bXNEABAJvvQU8WGOldMIi75BWdj8toKFZBMGaYxkP2AKH5jDlYm4oJSbYMcA8tojkgmrd9AHTZB6R7pECTYLN8OlsTU34IZBFZCTDckQc9Do7uqezDcvyAAMrW4dLNnvdKb7zjYXZCGczjZA3kY6ZCZB12ZCE2QIZBoG1Bpg9biL27VeN0qe'   #ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
VERIFY_TOKEN = 'simple-bot'   #VERIFY_TOKEN = os.environ['VERIFY_TOKEN']
bot = Bot (ACCESS_TOKEN)
model_fast = fasttext.FastText.load_model('cc.th.100.bin')
#loaded_model = pickle.load(open("thai_svm_100_joblib", 'rb'))

loaded_model = joblib.load('thai_svm_100_joblib.sav')

#We will receive messages that Facebook sends our bot at this endpoint 
@app.route("/", methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        """Before allowing people to message your bot, Facebook has implemented a verify token
        that confirms all requests that your bot receives came from Facebook.""" 
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    #if the request was not get, it must be POST and we can just proceed with sending a message back to user
    else:
        # get whatever message a user sent the bot
        output = request.get_json()
        for event in output['entry']:
          messaging = event['messaging']
          for message in messaging:
            if message.get('message'):
                #Facebook Messenger ID for user so we know where to send response back to
                recipient_id = message['sender']['id']
                if message['message'].get('text'):
                    response_sent_text = get_message()
                    print("")
                    print(message['message'].get('text'))
                    print("")
                    #print(get_vector(all_clean(message['message'].get('text'))))
                    clean = all_clean(message['message'].get('text'))
                    if len(clean) == 0 :
                        print("empty after clean")
                        print("nothing here")
                    else :
                        vec = get_vector(clean)
                        print(vec)
                        pre = loaded_model.predict(vec)
                        if pre == -1 :
                            send_message(recipient_id,response_sent_text)
                            print(-1)
                        else :
                            send_message(recipient_id,"ขอให้เป็นวันที่ดี")
                            print(1)
                    #print(all_clean(message['message'].get('text')))
                    #print(type(message['message'].get('text')))
                    #print("เป็นอะไร")
                    #print(loaded_model.shape_fit_)
                    #print(sklearn.__version__)
                    #print(loaded_model.predict(get_vector("อยากตาย"))[0])
                    #print(get_vector(all_clean(message['message'].get('text'))))
                    #pre = loaded_model.predict(vec)
                   
                    #if pre == -1 :
                    #    send_message(recipient_id,response_sent_text)
                    #    print(-1)
                       
                   # else :
                    #    send_message(recipient_id,"อืออออ")
                     #   print(1)
                    
                    
                    #loaded_model = pickle.load(open("thai_svm.sav", 'rb'))

                    #test = loaded_model.predict(vec)[0]
                    #print(test)
                    #print(loaded_model.predict(get_vector(all_clean(message['message'].get('text'))).reshape(1,-1))[0])
                #if user sends us a GIF, photo,video, or any other non-text item
                if message['message'].get('attachments'):
                    response_sent_nontext = get_message()
                    send_message(recipient_id, response_sent_nontext)
    return "Message Processed"


def verify_fb_token(token_sent):
    #take token sent by facebook and verify it matches the verify token you sent
    #if they match, allow the request, else return an error 
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


#chooses a random message to send to the user
def get_message():
    sample_responses = ["อีกไม่นานก็จะดีขึ้นและเธอจะผ่านมันไปได้", "ฉันอาจจะไม่เข้าใจเธอ แต่ฉันจะอยู่ข้างๆ เธอนะ", "เธอยังมีเวลาอีกมาก และฉันจะอยู่ข้างๆ เผื่อว่าจะช่วยอะไรเธอได้บ้าง", "อดทนไว้นะ เธอยังมีฉันอยู่ข้างๆ นะ", "เธอไม่ได้อยู่คนเดียวนะ", "ไม่มีใครตั้งใจให้เรื่องร้ายๆ เกิดขึ้นหรอก", "ฉันเห็นแล้วว่าเธอกำลังพยายาม มีอะไรที่ฉันพอจะช่วยเธอได้บ้าง", "ฉันจะกอดเธอไว้นะ"]
    # return selected item to the user
    return random.choice(sample_responses)
 
#uses PyMessenger to send response to user
def send_message(recipient_id, response):
    #sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"

#uses find vector of sentence

def get_vector(a) :
    #global vec_word_test
    vec_word_test = np.zeros(100)
    vec_allsent =[]
    word_token = word_tokenize(a,keep_whitespace=False)
    for j in word_token :
        vec_word = []
        #vec_word_test = np.zeros(100)
        if j in model_fast.words :
            vec_word_test = vec_word_test + model_fast.get_word_vector(j)
            # print("found : {}".format(j))
        else:
            vec_word_test = vec_word_test + np.zeros(100)
            # print("Not found : {}".format(j))
    vec_allsent.append(np.divide(vec_word_test,len(word_token))) 
    return vec_allsent

#clean emoji
def remove_emoji(string):
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002500-\U00002BEF"  # chinese char
                        u"\U00002702-\U000027B0"
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001f926-\U0001f937"
                        u"\U00010000-\U0010ffff"
                        u"\u2640-\u2642" 
                        u"\u2600-\u2B55"
                        u"\u200d"
                        u"\u23cf"
                        u"\u23e9"
                        u"\u231a"
                        u"\ufe0f"  # dingbats
                        u"\u3030"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

#clean text 
def tweet_cleaning (tweet):
  #1 Remove URL
   tweet = ' '.join(re.sub('https?://[A-Za-z0-9./]+','',tweet).split())
  #2 Removal of mention@ ,hastag# ( but keep text after #)
   tweet = ' '.join(re.sub('@[A-Za-z0-9_]+', '',tweet).split())
   tweet = ' '.join(re.sub('(#[ก-๙A-Za-z0-9]+)|(@[A-Za-z0-9]+)|(#)|(&amp;) ','',tweet).split())
  # #3 keep only alphabet
   tweet = ' '.join(re.sub('[^ก-๙A-Za-z]','', tweet).split())
  # #4 to String 
  #  tweet = tweet.replace(r'RT','')
  #  #5 
  #  tweet = tweet.strip()
  #  #6 
  #  tweet = tweet.lower()
   return tweet

def all_clean(a) :
    a = remove_emoji(a)
    a = tweet_cleaning(a)
    return a



#def get_predict(a) :
#    loaded_model = pickle.load(open("thai_svm.sav", 'rb'))
#    test = loaded_model.predict(np.zeros(300))
#    return None

if __name__ == "__main__":
    app.run()
