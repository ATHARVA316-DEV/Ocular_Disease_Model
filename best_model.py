import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k")

print("Path to dataset files:", path)


import os
path="/kaggle/input/ocular-disease-recognition-odir5k"
base_path = path 


# Show folder structure
# for root, dirs, files in os.walk(base_path):
#     print(root)
#     break

# !ls -R "$base_path"

import os

base_path = os.path.join(path, "ODIR-5K", "ODIR-5K")

labels_file = os.path.join(base_path, "data.xlsx")
train_images_folder = os.path.join(base_path, "Training Images")
test_images_folder = os.path.join(base_path, "Testing Images")


import pandas as pd

df = pd.read_excel(labels_file)
print(df.head())


import torch # <-- I've added this line to fix the error
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.transforms import functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, f1_score

# # --- Constants ---
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}")

# IMG_SIZE = 224
# BATCH_SIZE = 32
# LEARNING_RATE = 1e-4
# EPOCHS = 15 # Start with 10, increase if needed

# # Disease categories from your 'data.xlsx'
# CLASSES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']


# dual cnn 
#main model(the proper one )
# ================================================================
# ODIR-5K Dual CNN  - Robust Version with Native PyTorch
# ================================================================
!pip install --quiet timm openpyxl pillow scikit-learn

import os, warnings, torch, timm
import numpy as np, pandas as pd
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torchvision.transforms as T
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE, BATCH_SIZE, EPOCHS, LR = 384, 6, 25, 5e-5
CLASSES = ['N','D','G','C','A','H','M','O']
NUM_CLASSES = len(CLASSES)
WARMUP_EPOCHS = 3
print("Device:", DEVICE)

# ----------------- Load Data for DUAL EYES -----------------
BASE = "/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K"
df = pd.read_excel(os.path.join(BASE,"data.xlsx"))

def build_dual_df(df):
    rows=[]
    for _,r in df.iterrows():
        labels=[int(r[c]) for c in CLASSES]
        left_p=os.path.join(BASE,"Training Images",f"{r['ID']}_left.jpg")
        right_p=os.path.join(BASE,"Training Images",f"{r['ID']}_right.jpg")
        if os.path.exists(left_p) and os.path.exists(right_p):
            rows.append({"left_path":left_p,"right_path":right_p,"labels":labels})
    return pd.DataFrame(rows)

df_all=build_dual_df(df)
train_df,val_df=train_test_split(df_all,test_size=0.15,random_state=42,shuffle=True)
print(len(train_df),"train pairs,",len(val_df),"val pairs")

# ----------------- Native PyTorch Augmentations -----------------
class TrainTransform:
    def __init__(self, size=384):
        self.size = size
        self.normalize = T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        
    def __call__(self, img):
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            img = T.functional.hflip(img)
        
        # Random vertical flip
        if torch.rand(1) < 0.2:
            img = T.functional.vflip(img)
        
        # Color jitter
        if torch.rand(1) < 0.6:
            brightness = 1 + (torch.rand(1) - 0.5) * 0.6
            contrast = 1 + (torch.rand(1) - 0.5) * 0.6
            img = T.functional.adjust_brightness(img, brightness.item())
            img = T.functional.adjust_contrast(img, contrast.item())
        
        # Random rotation
        if torch.rand(1) < 0.6:
            angle = (torch.rand(1) - 0.5) * 24
            img = T.functional.rotate(img, angle.item())
        
        # Convert to tensor and normalize
        img = T.functional.to_tensor(img)
        img = self.normalize(img)
        
        return img

class ValTransform:
    def __init__(self, size=384):
        self.size = size
        self.normalize = T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        
    def __call__(self, img):
        img = T.functional.to_tensor(img)
        img = self.normalize(img)
        return img

# ----------------- Dual Eye Dataset -----------------
class DualEyeDataset(Dataset):
    def __init__(self,df,transform,size=384):
        self.df=df.reset_index(drop=True)
        self.transform=transform
        self.size=size
    
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self,i):
        r=self.df.iloc[i]
        
        # Load and resize left eye
        left_img=Image.open(r["left_path"]).convert('RGB')
        left_img=left_img.resize((self.size,self.size),Image.BILINEAR)
        left_img=self.transform(left_img)
        
        # Load and resize right eye
        right_img=Image.open(r["right_path"]).convert('RGB')
        right_img=right_img.resize((self.size,self.size),Image.BILINEAR)
        right_img=self.transform(right_img)
        
        y=torch.tensor(r["labels"],dtype=torch.float32)
        return left_img,right_img,y

train_ds=DualEyeDataset(train_df,TrainTransform(IMG_SIZE),IMG_SIZE)
val_ds=DualEyeDataset(val_df,ValTransform(IMG_SIZE),IMG_SIZE)

label_arr=np.stack(train_df["labels"].values)
pos_counts=label_arr.sum(axis=0)
neg_counts=len(label_arr)-pos_counts
pos_weights=neg_counts/np.maximum(pos_counts,1)
weights=1.0/(label_arr.sum(axis=1)+1e-3)
sampler=WeightedRandomSampler(weights,len(weights),replacement=True)

train_dl=DataLoader(train_ds,batch_size=BATCH_SIZE,sampler=sampler,num_workers=2,pin_memory=True)
val_dl=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)

# ----------------- DUAL CNN MODEL -----------------
class DualCNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Shared backbone for both eyes
        self.backbone=timm.create_model("convnext_small",pretrained=True,num_classes=0)
        n=self.backbone.num_features
        
        # Individual eye processing branches
        self.left_branch=torch.nn.Sequential(
            torch.nn.Linear(n,384),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3)
        )
        self.right_branch=torch.nn.Sequential(
            torch.nn.Linear(n,384),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3)
        )
        
        # Cross-attention between eyes
        self.cross_attn=torch.nn.MultiheadAttention(384,num_heads=8,dropout=0.2,batch_first=True)
        
        # Fusion head
        self.fusion_head=torch.nn.Sequential(
            torch.nn.Linear(384*3,512),  # left + right + fused
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512,256),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256,NUM_CLASSES)
        )
    
    def forward(self,left,right):
        # Extract features from both eyes using shared backbone
        left_feat=self.backbone(left)
        right_feat=self.backbone(right)
        
        # Process through individual branches
        left_proc=self.left_branch(left_feat)
        right_proc=self.right_branch(right_feat)
        
        # Cross-attention: let each eye attend to the other
        left_attn=left_proc.unsqueeze(1)
        right_attn=right_proc.unsqueeze(1)
        
        # Attend left to right
        left_attended,_=self.cross_attn(left_attn,right_attn,right_attn)
        left_attended=left_attended.squeeze(1)
        
        # Attend right to left
        right_attended,_=self.cross_attn(right_attn,left_attn,left_attn)
        right_attended=right_attended.squeeze(1)
        
        # Fuse information
        fused=(left_attended+right_attended)/2
        
        # Concatenate all representations
        combined=torch.cat([left_proc,right_proc,fused],dim=1)
        
        return self.fusion_head(combined)

model=DualCNNModel().to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ----------------- Loss Functions -----------------
class FocalLoss(torch.nn.Module):
    def __init__(self,alpha=0.25,gamma=2):
        super().__init__();self.alpha=alpha;self.gamma=gamma
    def forward(self,x,y):
        bce=F.binary_cross_entropy_with_logits(x,y,reduction='none')
        pt=torch.exp(-bce)
        focal=(1-pt)**self.gamma*bce
        return (self.alpha*y*focal+(1-self.alpha)*(1-y)*focal).mean()

focal=FocalLoss(alpha=0.3,gamma=2.5)
bce_weighted=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights,device=DEVICE).float())

def hybrid_loss(x,y): 
    return 0.5*bce_weighted(x,y)+0.5*focal(x,y)

# ----------------- Optimizer -----------------
opt=torch.optim.AdamW([
    {"params":model.backbone.parameters(),"lr":LR*0.1},
    {"params":model.left_branch.parameters(),"lr":LR*0.5},
    {"params":model.right_branch.parameters(),"lr":LR*0.5},
    {"params":model.cross_attn.parameters(),"lr":LR*0.7},
    {"params":model.fusion_head.parameters(),"lr":LR}
],weight_decay=2e-5)

def get_lr(epoch):
    if epoch<WARMUP_EPOCHS: return (epoch+1)/WARMUP_EPOCHS
    return 0.5*(1+np.cos(np.pi*(epoch-WARMUP_EPOCHS)/(EPOCHS-WARMUP_EPOCHS)))

sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=get_lr)
scaler=torch.cuda.amp.GradScaler()

# ----------------- Training -----------------
best_f1=0
accum_steps=2

for epoch in range(1,EPOCHS+1):
    model.train();tloss,allp,alll=0,[],[]
    pbar=tqdm(train_dl,desc=f"Epoch {epoch}/{EPOCHS}")
    opt.zero_grad()
    
    for step,(left,right,y) in enumerate(pbar):
        left,right,y=left.to(DEVICE),right.to(DEVICE),y.to(DEVICE)
        
        with torch.cuda.amp.autocast():
            out=model(left,right)
            loss=hybrid_loss(out,y)/accum_steps
        
        scaler.scale(loss).backward()
        
        if (step+1)%accum_steps==0 or step==len(train_dl)-1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt);scaler.update();opt.zero_grad()
        
        tloss+=loss.item()*left.size(0)*accum_steps
        allp.append(torch.sigmoid(out).detach().cpu().numpy())
        alll.append(y.cpu().numpy())
        pbar.set_postfix({"loss":f"{loss.item()*accum_steps:.4f}"})
    
    sched.step()
    allp,alll=np.vstack(allp),np.vstack(alll)
    preds=(allp>0.5).astype(int)
    trf1=f1_score(alll,preds,average='micro')
    
    # Validation
    model.eval();vloss,vlp,vll=0,[],[]
    with torch.no_grad():
        for left,right,y in val_dl:
            left,right,y=left.to(DEVICE),right.to(DEVICE),y.to(DEVICE)
            with torch.cuda.amp.autocast():
                o=model(left,right)
            vloss+=hybrid_loss(o,y).item()*left.size(0)
            vlp.append(torch.sigmoid(o).cpu().numpy())
            vll.append(y.cpu().numpy())
    
    vlp,vll=np.vstack(vlp),np.vstack(vll)
    val_preds=(vlp>0.5).astype(int)
    vf1=f1_score(vll,val_preds,average='micro')
    vf1m=f1_score(vll,val_preds,average='macro')
    print(f"Train F1µ {trf1:.3f} | Val F1µ {vf1:.3f} | Val F1M {vf1m:.3f}")
    
    # Per-class F1
    f1s=[f1_score(vll[:,i],val_preds[:,i],zero_division=0) for i in range(NUM_CLASSES)]
    print("Per-class F1:",{CLASSES[i]:round(f1s[i],3) for i in range(NUM_CLASSES)})
    
    if vf1>best_f1:
        best_f1=vf1
        torch.save(model.state_dict(),"best_dual_f1_model.pth")
        print(f"✅ Saved new best (F1µ={best_f1:.3f})")

print("\n🎯 Best Validation F1-micro:",best_f1)



