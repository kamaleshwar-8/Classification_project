import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
import pandas as pd
face_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
# img=cv2.imread('test_img/virat1.jpg')
# print(img.shape)

# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# plt.imshow(gray,cmap='gray')
# face_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
# face=face_cascade.detectMultiScale(gray,1.3,5)
# (x,y,w,h)=face[0]
# face_img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# for(x,y,w,h) in face:
#     face_img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray=gray[y:y+h,x:x+w]
#     roi_color=face_img[y:y+h,x:x+w]
#     eyes=eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in  eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# plt.imshow(roi_color,cmap='gray')
def get_crimg(path):
    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,h,w) in face:
        roi_gray=gray[y:y+h,x:x+h]
        roi_color=img[y:y+h,x:x+h]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        if len(eyes)>=2:
            return roi_color
        
crop_img=get_crimg('test_img/virat1.jpg')
# plt.imshow(crop_img)
path_to_data='./dataset/'
path_dir="./dataset./cropped/"
import os
img_dirs=[]
for i in os.scandir(path_to_data):
    if i.is_dir():
        img_dirs.append(i.path)
import shutil
if os.path.exists(path_dir):
    shutil.rmtree(path_dir)
os.mkdir(path_dir)

crop_img_dir=[]
name_dict={}

for imgs in img_dirs:
    names=imgs.split('/')[-1]
    name_dict[names]=[]
    count=0
    for i in os.scandir(imgs):
        roi_color=get_crimg(i.path)
        if roi_color is not None:
            cropped_folder=path_dir + names
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                crop_img_dir.append(cropped_folder)
                print('generating in :',cropped_folder)
            cropped_fname=names+str(count)+'.png'
            cropped_fpath=cropped_folder+'/'+cropped_fname
            count+=1
            cv2.imwrite(cropped_fpath,roi_color)
            name_dict[names].append(cropped_fpath)
def w2d(img,mode='haar',level=1):
    imarray=img
    imarray=cv2.cvtColor(imarray,cv2.COLOR_BGR2GRAY)
    imarray=np.float32(imarray)
    imarray/=255;
    coeff=pywt.wavedec2(imarray,mode,level=level)
    coeffh=list(coeff)
    coeffh[0]*=0;
    imarrayh=pywt.waverec2(coeffh,mode)
    imarrayh*=255
    imarrayh=np.uint8(imarrayh)
    return imarrayh

im_har=w2d(crop_img,'db1',5)
# plt.imshow(im_har,cmap='gray')

class_dict={}
count=0
for n in name_dict.keys():
    class_dict[n]=count
    count+=1
print(class_dict)
x=[]
y=[]

for name ,trainingfile in name_dict.items():
    for trimg in trainingfile:
        img=cv2.imread(trimg)
        if img is None:
            continue
        sca_rawimg=cv2.resize(img,(32,32))
        img_har=w2d(img,'db1',5)
        sca_img_har=cv2.resize(img_har,(32,32))
        com_img=np.vstack((sca_rawimg.reshape(32*32*3,1),sca_img_har.reshape(32*32,1)))
        x.append(com_img)
        y.append(class_dict[name])
print(len(x[0]))
x=np.array(x).reshape(len(x),4096).astype(float)
print(x.shape)
# from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
# pipe=Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel='rbf',C=50))])
# pipe.fit(x_train,y_train)
# print(pipe.score(x_test,y_test))
# print(classification_report(y_test, pipe.predict(x_test)))
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
model_param={
    'svm':{
        'model':svm.SVC(gamma='auto',probability=True),
        'params':{
            'svc__C':[1,5,10,50,75,100,1000],
            'svc__kernel':['rbf','linear']
            }
        },
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'randomforestclassifier__n_estimators':[1,5,7,10,15,20]
            }
        },
    'log_reg':{
        'model':LogisticRegression(max_iter=200),
        'params':{
            'logisticregression__C':[1,2,3,4,5,6,8,9]
            }
        }
    }

score=[]
best_est={}
for a,mp in model_param.items():
    pipe=make_pipeline(StandardScaler(),mp['model'])
    clf=GridSearchCV(pipe, mp['params'],cv=5,return_train_score=False)
    clf.fit(x_train,y_train)
    score.append({
        'model':a,
        'best_score':clf.best_score_,
        'best_param':clf.best_params_})
    best_est[a]=clf.best_estimator_
df=pd.DataFrame(score,columns=['model','best_score','best_param'])
best_clf=best_est['log_reg']
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,best_clf.predict(x_test))
import seaborn as sns
sns.heatmap(cm)
plt.show()
import joblib
joblib.dump(best_clf,'saved_model.pk1')
import json
with open("class_dictionary.json","w")as f:
    f.write(json.dumps(class_dict))