import numpy as np
import os
from face_recognition.face_utils import register_face
REG_DIRS="data/registered_faces"
REF_DIRS="data/ref"
ref=os.listdir("data/ref")
for i in ref:
    x=i.lower()
    t=x.endswith((".jpg",".jpeg",".png"))
    if(t):
        y=os.path.splitext(i)
        username=i.replace(y[1],"")
        path=os.path.join(REF_DIRS,i)
        try:
            o=username+".npy"
            spath=os.path.join(REG_DIRS,o)
            if(os.path.exists(spath)):
                print(f"[INFO]-{username} already exists")
            else:
                x=register_face(username,path)
        except Exception as e:
            print(f"[INFO] failed - {username} not registered")
            print(f"error occured:{e}")
    

    
