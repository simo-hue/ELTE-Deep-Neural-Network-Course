import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# implement your solution BELOW

# ---- 1. Setup (fresh model every time this cell runs) ----
torch.manual_seed(911)

# Instance of the specific object ( with our inputs )
n_inputs = x_tr.shape[1]
reg_model = RegressionMLP(n_inputs)

# MSE is our loss function
loss_fn = nn.MSELoss()

# Setting what the optimizer can adjust
optimizer = torch.optim.Adam(reg_model.parameters(), lr=0.001)


N_EPOCHS = 55

train_losses, val_losses = [], []

# ---- 2. Training loop ----
for epoch in range(N_EPOCHS):
    # need to swithc the training flag
    reg_model.train()
    
    # I want to save them
    batch_losses = []
    
    for xb, yb in dataloader_reg_train:
        # 1. forward
        pred = reg_model(xb)                       
        
        # 2. loss
        loss = loss_fn(pred, yb)        
        
        # 3. gradients accumulate, so reset them before each new backward()
        optimizer.zero_grad()                  
        
        # 4. compute gradients -> Backward step
        loss.backward()           
        
        # 5. update weights
        optimizer.step()
        
        # Save the losses                  
        batch_losses.append(loss.item())
        
    train_losses.append(np.mean(batch_losses))

    # validation -> Turn off the training flag
    reg_model.eval()
    
    batch_losses = []
    
    # I'm not training
    with torch.no_grad():
        for xb, yb in dataloader_reg_val:
            batch_losses.append(loss_fn(reg_model(xb), yb).item())
            
    val_losses.append(np.mean(batch_losses))

    if epoch % 9 == 0:
        print(f"epoch {epoch:3d}   train {train_losses[-1]:7.3f}   val {val_losses[-1]:7.3f}")

# ---- 3. Plot ----
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.ylim(0, 15)                         
plt.legend()
plt.show()

# ---- 4. Test MSE on the FULL test set ----
reg_model.eval()
preds, trues = [], []

with torch.no_grad():
    for xb, yb in dataloader_reg_test:
        preds.append(reg_model(xb))
        trues.append(yb)
        
preds = torch.cat(preds, axis=0)
trues = torch.cat(trues, axis=0)
test_mse = loss_fn(preds, trues).item()

print("test MSE:", test_mse)

# ---- 5. Check the example predictions ----
for p, t in zip(preds[:10, 0], trues[:10, 0]):
    print(f"predicted {p.item():5.2f}   true {t.item():4.0f}")

print()
    
# implement your solution ABOVE

tester.test('reg_model_learning', test_mse)