import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,KFold
from torch.utils.data import TensorDataset, DataLoader
import random
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt

#chuli一下随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

#导入数据并且处理一下吧
tt=pd.read_csv("titanic_train.csv")
tt["familyAll"]=tt["SibSp"]+tt["Parch"]+1

def comeon_name(name):
    if "Mr." in name:
        return "Mr."
    elif "Mrs." in name:
        return "Mrs."
    elif "Miss." in name:
        return "Miss."
    elif "Master." in name:
        return "Master."
    else:
        return "Other"

tt["identify"]=tt["Name"].apply(comeon_name)

#删除我不需要的列
tt.drop(columns=["PassengerId","Name","Ticket","Cabin","SibSp","Parch"],axis=1,inplace=True)

insame_weight = ["Pclass", "Age", "Fare","familyAll"]    #number good
same_weight = ["Sex", "Embarked","identify"]             #要独热编码的

X = tt[insame_weight+same_weight].copy()
y = tt["Survived"]

def get_preprocessor():
    return ColumnTransformer([
        ("same", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(sparse_output=False))
        ]), same_weight),
        ("insame", Pipeline([
            ("fill", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler())
        ]), insame_weight),
    ])

X_k, X_test, y_k, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)

#定义一下网络吧
level_1 = 28
level_2 = 14
drop_num=0.28
lr_num=0.01
class Net(nn.Module):
     def __init__(self,fea):
         super().__init__()
         self.net = nn.Sequential(
             nn.Linear(fea,level_1),
             nn.BatchNorm1d(level_1),
             nn.ReLU(),
             nn.Dropout(drop_num),
             nn.Linear(level_1,level_2),
             nn.BatchNorm1d(level_2),
             nn.ReLU(),
             nn.Dropout(drop_num),
             nn.Linear(level_2,1)
         )
     def forward(self,x):
         return self.net(x)

# KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_val_acc_list = []
fold_best_weights = []
fold_loss_records = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_k)):
    print(f"\n========== BRo now Fold {fold+1} ==========")
    X_fold_train, X_fold_val = X_k.iloc[train_idx], X_k.iloc[val_idx]
    y_fold_train, y_fold_val = y_k.iloc[train_idx], y_k.iloc[val_idx]

    prep = get_preprocessor()
    X_fold_train = prep.fit_transform(X_fold_train)
    X_fold_val = prep.transform(X_fold_val)

    X_fold_train = torch.from_numpy(X_fold_train).float()
    y_fold_train = torch.from_numpy(y_fold_train.values).float().unsqueeze(1)
    X_fold_val = torch.from_numpy(X_fold_val).float()
    y_fold_val = torch.from_numpy(y_fold_val.values).float().unsqueeze(1)

    model = Net(X_fold_train.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=lr_num)
    loss = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(TensorDataset(X_fold_train,y_fold_train), batch_size=32, shuffle=True)

    best_val_loss = float("inf")
    best_w = None
    patience = 10
    stop_cnt = 0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(60):
        model.train()
        total_train_loss = 0
        total_correct = 0
        total_samples = 0
        for bx, by in train_loader:
            pred = model(bx)
            Loss = loss(pred, by)
            optim.zero_grad()
            Loss.backward()
            optim.step()
            total_train_loss += Loss.item()
            # 训练集acc计算
            pred_bin = (pred > 0).float()
            total_correct += (pred_bin == by).sum().item()
            total_samples += bx.shape[0]
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = total_correct / total_samples
        #print(f"Epoch {epoch}, avg_train_loss: {avg_train_loss:.4f}")
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_acc)

        # val
        model.eval()
        with torch.no_grad():
            val_logits = model(X_fold_val)
            val_loss = loss (val_logits, y_fold_val)
            #print(f"Epoch {epoch}, val loss: {val_loss:.4f}")
            val_pred = (val_logits > 0).float()
            val_acc = (val_pred == y_fold_val).sum().item() / y_fold_val.shape[0]
        val_loss_history.append(val_loss.item())
        val_acc_history.append(val_acc)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_w = model.state_dict()
            stop_cnt = 0
        else:
            stop_cnt += 1
            if stop_cnt >= patience:
                print(f"Fold {fold + 1} early stop @ epoch {epoch}")
                break
    # 保存当前折loss、acc记录
    fold_loss_records.append({
        "train":train_loss_history,
        "val":val_loss_history,
        "train_acc":train_acc_history,
        "val_acc":val_acc_history
    })

    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        val_pred = (model(X_fold_val) > 0).float()
        val_acc = (val_pred == y_fold_val).sum().item() / y_fold_val.shape[0]
    fold_val_acc_list.append(val_acc)
    print(f"Fold {fold + 1} best val acc: {val_acc:.4f}")

# K折汇总
cv_mean_acc, cv_std_acc = np.mean(fold_val_acc_list), np.std(fold_val_acc_list)
print("\n=====5折result_all=====")
print(f"第一层参数个数:{level_1},第二层参数个数：{level_2},drop百分比：{drop_num},学习率：{lr_num}")
print(f"每折acc列表：{[round(x, 4) for x in fold_val_acc_list]}")
print(f"5折平均acc:{cv_mean_acc:.4f}, 标准差:{cv_std_acc:.4f}")

prep_final = get_preprocessor()
prep_final.fit(X_k)

X_test_final = prep_final.transform(X_test)
X_test_tensor = torch.from_numpy(X_test_final).float()
y_test_tensor = torch.from_numpy(y_test.values).float().unsqueeze(1)


X_final_raw = prep_final.transform(X_k)
y_final_raw = y_k.values
X_train_final_raw, X_val_final_raw, y_train_final_raw, y_val_final_raw = train_test_split(
    X_final_raw, y_final_raw, test_size=0.2, random_state=42, shuffle=True, stratify=y_final_raw
)

#转为tensor
X_train_final = torch.from_numpy(X_train_final_raw).float()
y_train_final = torch.from_numpy(y_train_final_raw).float().unsqueeze(1)
X_val_final = torch.from_numpy(X_val_final_raw).float()
y_val_final = torch.from_numpy(y_val_final_raw).float().unsqueeze(1)

train_dataset = TensorDataset(X_train_final, y_train_final)
val_dataset = TensorDataset(X_val_final, y_val_final)
full_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

final_model = Net(X_train_final.shape[1])
opt = torch.optim.Adam(final_model.parameters(), lr=lr_num)
loss_fn = nn.BCEWithLogitsLoss()
best_final_loss, best_final_w, cnt = float("inf"), None, 0

final_train_loss = []
final_train_acc = []
final_val_loss = []
final_val_acc = []

for epoch in range(60):
    final_model.train()
    total_tr_loss = 0
    total_correct_tr = 0
    total_samp_tr = 0
    for bx, by in full_loader:
        pred = final_model(bx)
        loss = loss_fn(pred, by)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_tr_loss += loss.item()
        pred_bin = (pred > 0).float()
        total_correct_tr += (pred_bin == by).sum().item()
        total_samp_tr += bx.shape[0]
    avg_tr_loss = total_tr_loss / len(full_loader)
    avg_tr_acc = total_correct_tr / total_samp_tr

    final_model.eval()
    total_v_loss = 0
    total_correct_v = 0
    total_samp_v = 0
    with torch.no_grad():
        for bx_val, by_val in val_loader:
            pred_v = final_model(bx_val)
            loss_v = loss_fn(pred_v, by_val)
            total_v_loss += loss_v.item()
            pred_bin_v = (pred_v > 0).float()
            total_correct_v += (pred_bin_v == by_val).sum().item()
            total_samp_v += bx_val.shape[0]
    avg_val_loss = total_v_loss / len(val_loader)
    avg_val_acc = total_correct_v / total_samp_v

    final_train_loss.append(avg_tr_loss)
    final_train_acc.append(avg_tr_acc)
    final_val_loss.append(avg_val_loss)
    final_val_acc.append(avg_val_acc)

    # 早停
    if avg_val_loss < best_final_loss:
        best_final_loss = avg_val_loss
        best_final_w = final_model.state_dict()
        cnt = 0
    else:
        cnt += 1
        if cnt >= 10:
            print(f"Final model early stop epoch {epoch}")
            break

final_model.load_state_dict(best_final_w)
final_model.eval()
with torch.no_grad():
    logits = final_model(X_test_tensor)
    test_pred = (logits > 0).float()
    holdout_test_acc = (test_pred == y_test_tensor).sum().item() / len(y_test_tensor)
    holdout_loss = loss_fn(logits, y_test_tensor).item()
print(f"\n===== Hold-out测试集评估 =====")
print(f"Hold-out Test Acc = {holdout_test_acc:.4f}")
print(f"Hold-out Test Loss = {holdout_loss:.4f}")

# save
torch.save({"model_state_dict": best_final_w}, "titanic_final_best.pth")#保存网路权重
joblib.dump(prep_final, "titanic_prep_final.pkl")#存预处理的prep转换

#终于画图
plt.rcParams['font.sans-serif'] = ['SimHei']#设置中文黑体
# 1.来张5折图
fig, axes = plt.subplots(3,2,figsize=(12,14))
axes = axes.flatten()
for i, rec in enumerate(fold_loss_records):
    ax = axes[i]
    ax.plot(rec["train"], label="Train Loss")
    ax.plot(rec["val"], label="Val Loss")
    ax.set_title(f"Fold{i+1} Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)

axes[5].axis("off")
plt.tight_layout()
plt.savefig("../docs/kfold_loss_all.png",dpi=300)
plt.close()

#2.最终模型损失趋势图
plt.figure(figsize=(7,5))
plt.plot(final_train_loss,label="Train Loss")
plt.plot(final_val_loss,label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Final Model Loss Curve")
plt.legend()
plt.grid(alpha=0.26)
plt.tight_layout()
plt.savefig("../docs/final_model_loss.png",dpi=300)
plt.close()

#3.准确率
plt.figure(figsize=(7,5))
plt.plot(final_train_acc,label="Train Acc")
plt.plot(final_val_acc,label="Val Acc")
plt.axhline(y=holdout_test_acc,color="orange",linestyle="-",label="Holdout Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Final Model Accuracy Curve")
plt.legend()
plt.grid(alpha=0.27)
plt.tight_layout()
plt.savefig("../docs/final_model_acc.png",dpi=300)
plt.close()

# 图4：原有5折准确率散点图
plt.figure(figsize=(7, 5))
plt.scatter(np.arange(1, 6),fold_val_acc_list,s=90,c="#1f77b4")
plt.axhline(y=cv_mean_acc, c='r',linestyle="--",label=f"5折平均acc={cv_mean_acc:.4f}")
plt.xlabel("Fold序号")
plt.ylabel("验证集准确率")
plt.title("5折交叉验证每折Val准确率")
plt.legend()
plt.xticks(np.arange(1, 6))
plt.tight_layout()
plt.savefig("../docs/kfold_val_acc.png", dpi=300)
plt.close()

def predict_new(person):
    tt_new = pd.DataFrame([person])
    tt_new["familyAll"] = tt_new["SibSp"] + tt_new["Parch"] + 1
    tt_new["identify"] = comeon_name(tt_new["Name"])
    tt_new = tt_new[["Sex","Embarked","identify","Pclass","Age","Fare", "familyAll"]]
    prep_load = joblib.load("titanic_prep_final.pkl")
    x_1 = prep_load.transform(tt_new)
    x_tensor = torch.from_numpy(x_1).float()
    weight = torch.load("titanic_final_best.pth")#字典
    model = Net(x_tensor.shape[1])
    model.load_state_dict(weight["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logit = model(x_tensor)
    return 1 if logit.item() > 0 else 0

person={
    "Pclass":1,
    "Name":"Mrs. Lily",
    "Sex":"female",
    "Age":30,
    "SibSp":1,
    "Parch":0,
    "Fare":80,
    "Embarked":"S"
    }
res = predict_new(person)
print(f"\n新样本预测结果（1幸存，0遇难）：{res}")
