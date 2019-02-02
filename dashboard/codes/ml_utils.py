import numpy as np
from keras.models import model_from_json
import librosa
import pandas as pd
from django.conf import settings
import os
from pydub import AudioSegment
labels={0 :' female_angry', 1 : 'female_calm', 2 : 'female_fearful', 3 : 'female_happy ', 4 : 'female_sad ',
5 : 'male_angry ', 6 : 'male_calm', 7 : 'male_fearful', 8 : 'male_happy', 9 : 'male_sad'} 

def load_model():
    with open(os.path.join(settings.BASE_DIR, 'dashboard/static/model.json'), 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(os.path.join(settings.BASE_DIR, 'dashboard/static/Emotion_Voice_Detection_Model.h5'))
    return model

def dir_predict(filename):
        filename = os.path.join(settings.BASE_DIR, filename[1:])
        print(filename)
        if filename[-3:]=="mp3" :
            sound = AudioSegment.from_mp3(filename)
            sound.export(filename[:-3]+"wav", format="wav")
            filename = filename[:-3]+"wav"
            
        model = load_model()
        x, sample_rate = librosa.load(filename, res_type='kaiser_fast',sr=22050*2) 
        print(len(x))
        ansarray=[]
        for i in range(0,len(x)-(len(x)%(22050*5)),22050*5):
            X=x[i:i+22050*5]
            print(i+22050*5)
            print(sample_rate)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate, n_mfcc=13),axis=0)
            feature = mfccs
            livedf2= pd.DataFrame(data=feature)
            livedf2 = livedf2.stack().to_frame().T
            inp_x= np.expand_dims(livedf2, axis=2)
            pred = model.predict(inp_x, batch_size=32, verbose=1)
            maxpos=np.argmax(pred,axis=1)
            male=0
            if maxpos>=5:
                male=1
            
    #         return dict(zip(labels.values(),pred[0]))
            oldmax = np.max(pred[0])
            oldmin = np.min(pred[0])
            oldrange = oldmax-oldmin
            pred = np.exp(pred-np.max(pred))/np.exp(pred-np.max(pred)).sum()*100
            # pred = (pred - oldmin)/oldrange
            ansarray.append(pred)
        return ansarray
