import torch
import torch.nn.functional as F
import copy
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from opacus import PrivacyEngine

def train_model(model, train_loader, optimizer, criterion, device, epochs=10, val_loader=None, patience=None, mixup_alpha=0.0):
    model.train()
    best_state = None
    best_val = None
    no_improve = 0
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            if mixup_alpha > 0.0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(images.size(0), device=device)
                mixed_images = lam * images + (1 - lam) * images[idx]
                mixed_labels = lam * labels + (1 - lam) * labels[idx]
                inputs = mixed_images
                targets = mixed_labels
            else:
                inputs = images
                targets = labels
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if val_loader is not None:
            model.eval()
            vals = []
            with torch.no_grad():
                for vimages, vlabels in val_loader:
                    vimages = vimages.to(device)
                    vlabels = vlabels.to(device)
                    voutputs = model(vimages)
                    vloss = criterion(voutputs, vlabels)
                    vals.append(vloss.item())
            cur_val = float(np.mean(vals)) if len(vals) > 0 else None
            if cur_val is not None:
                if best_val is None or cur_val < best_val:
                    best_val = cur_val
                    best_state = copy.deepcopy(model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if patience is not None and no_improve >= patience:
                        break
            model.train()
    if best_state is not None:
        model.load_state_dict(best_state)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return accuracy, f1

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _logits_from_probs(p):
    eps = 1e-6
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p / (1 - p))

def fit_temperature(model, val_loader, device, max_iter=100):
    model.eval()
    t_param = torch.tensor(0.0, device=device, requires_grad=True)
    opt = torch.optim.Adam([t_param], lr=0.01)
    for _ in range(max_iter):
        losses = []
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                probs = model(images)
                logits = _logits_from_probs(probs)
            T = F.softplus(t_param) + 1e-6
            scaled = torch.sigmoid(logits / T)
            loss = F.binary_cross_entropy(scaled, labels)
            losses.append(loss)
        if len(losses) == 0:
            break
        total = torch.stack(losses).mean()
        opt.zero_grad()
        total.backward()
        opt.step()
    T_final = float(F.softplus(t_param).item() + 1e-6)
    return T_final

def train_student_with_distillation(student, teacher, train_loader, optimizer, device, epochs=10, alpha=0.5, temperature=2.0):
    student.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                t_probs = teacher(images)
                t_logits = _logits_from_probs(t_probs)
                t_soft = torch.sigmoid(t_logits / temperature)
            optimizer.zero_grad()
            s_probs = student(images)
            s_logits = _logits_from_probs(s_probs)
            s_soft = torch.sigmoid(s_logits / temperature)
            distill_loss = F.mse_loss(s_soft, t_soft)
            ce_loss = F.binary_cross_entropy(s_probs, labels)
            loss = alpha * distill_loss + (1 - alpha) * ce_loss
            loss.backward()
            optimizer.step()
