import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

##########################
# 构造一份“有真实规律”的数据
##########################
torch.manual_seed(0)

# 假设真实模型 y = Wx + b
true_W = torch.tensor([[2.0, -3.0, 1.0]]).T  # (3,1)
true_b = torch.tensor([0.5])

# 构造训练数据
N = 200
x = torch.randn(N, 3)
noise = 0.1 * torch.randn(N, 1)
y = x @ true_W + true_b + noise

###########################
# 定义模型、损失函数、优化器
###########################
model = nn.Linear(3, 1)
criterion = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-2,
    weight_decay=1e-2,
)

###########################
# 多 epoch 训练，并记录 loss
###########################
num_epochs = 200
loss_history = []

for epoch in range(num_epochs):
    # ===== 1. 清空梯度 =====
    optimizer.zero_grad()

    # ===== 2. 前向传播 =====
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # ===== 3. 反向传播 =====
    loss.backward()

    # ===== 4. 参数更新 =====
    optimizer.step()

    # ===== 5. 记录 loss =====
    loss_history.append(loss.item())

    # 打印日志
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1:02d}/{num_epochs}] | Loss = {loss.item():.6f}")


plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.show()
