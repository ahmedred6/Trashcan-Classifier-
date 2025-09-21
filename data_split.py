import os, shutil, random
dataset_dir = "garbage_classification"
output_dir = "garbage_split"

train_split = 0.7
val_split = 0.2
test_split = 0.1

for split in ["train","val","test"]:
    if not os.path.exists(os.path.join(output_dir,split)):
        os.makedirs(os.path.join(output_dir,split))

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(output_dir,split,class_name)
    
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

for class_name in os.listdir(dataset_dir):    
    images = os.listdir(os.path.join(dataset_dir,class_name))
    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total*train_split)
    n_val = int(n_total*val_split)
    
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]
    
    for img in train_imgs:
        shutil.copy(os.path.join(dataset_dir,class_name,img),os.path.join(output_dir,"train",class_name,img))    
    
    for img in val_imgs:
        shutil.copy(os.path.join(dataset_dir,class_name,img),os.path.join(output_dir,"val",class_name,img)) 
    
    for img in test_imgs:
        shutil.copy(os.path.join(dataset_dir,class_name,img),os.path.join(output_dir,"test",class_name,img))    