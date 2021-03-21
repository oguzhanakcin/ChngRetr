import os,random,json
import torch
import cv2
import numpy as np

def get_img_locs(dir, ext=".tif"):
    imgs_loc = []
    for (dirpath, _, filenames) in os.walk(dir):
        for file in filenames:
            if file.endswith(ext):
                imgs_loc.append(os.path.join(dirpath, file))
    return imgs_loc

def create_test_trainset(t1_locs,train_ratio,out_dir):
    s_size = len(t1_locs)
    train_t1 = random.sample(t1_locs,int(s_size*train_ratio))
    test_t1 = [i for i in change_images_t1 if i not in train_t1]

    with open(out_dir+"/"+"train.json","w") as trainfile:
        json.dump(train_t1,trainfile,indent=4)
    with open(out_dir+"/"+"test.json", "w") as testfile:
        json.dump(test_t1,testfile,indent=4)

def read_augment_images(imgs):
    for img_loc in imgs:
        img1 = cv2.imread(img_loc)
        img2 = cv2.imread(img_loc.replace("/t1/","/t2/"))
        aug = random.random()
        if  aug < 0.2 :
            img1 = cv2.rotate(img1,rotateCode=cv2.ROTATE_90_CLOCKWISE)
            img2 = cv2.rotate(img2, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        elif aug < 0.4 :
            img1 = cv2.rotate(img1, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
            img2 = cv2.rotate(img2, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif aug < 0.6:
            img1 = cv2.rotate(img1, rotateCode=cv2.ROTATE_180)
            img2 = cv2.rotate(img2, rotateCode=cv2.ROTATE_180)
        elif aug < 0.8:
            img1 = cv2.flip(img1,0)
            img2 = cv2.flip(img2,0)
        else :
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
    t1 = ((img1.astype(np.float)/255.0)*2)-1
    t2 = ((img2.astype(np.float)/255.0)*2)-1

    return t1,t2
#create_test_trainset(change_images_t1,0.8,".")

with open("test.json") as json_file:
    data = json.load(json_file)


dataload = torch.utils.data.DataLoader(data,shuffle=True,batch_size=1)



