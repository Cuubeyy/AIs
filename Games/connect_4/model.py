import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(input_size, hidden_size).to("cuda")
        self.linear2 = nn.Linear(hidden_size, hidden_size).to("cuda")
        self.linear3 = nn.Linear(hidden_size, hidden_size).to("cuda")
#        self.linear4 = nn.Linear(hidden_size, hidden_size).to("cuda")
        self.linear_last = nn.Linear(hidden_size, output_size).to("cuda")

    def forward(self, x):
        x = x.to("cuda")
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
#        x = F.relu(self.linear4(x))
        x = self.linear_last(x)
        return x

    def save(self, file_name='model.pth', model='./model'):
        model_folder_path = model
        if not os.path.exists(model_folder_path):
            files = os.listdir('path')
            file_name = file_name + str(len(files))
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, move, reward, next_state, done):
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            move = torch.unsqueeze(move, 0)
            #reward = torch.unsqueeze(reward, 0)
            done = (done,)
        else:
            state = torch.stack(state)
            next_state = torch.stack(next_state)
            move = torch.stack(move)

        predicted_move = self.model(state)

        target = predicted_move.clone()
        prev_q_new = -10
        for idx in range(len(done) - 1, -1, -1):
            q_new = reward[idx]
            if not done[idx]:
                emil_award = -prev_q_new * self.gamma
                q_new = emil_award - self.gamma * max(torch.max(self.model(next_state[idx])), prev_q_new) / 10
            this_move = torch.argmax(move[idx]).item()
            target[idx][this_move] = q_new
            prev_q_new = q_new


        target = target.to("cuda")
        predicted_move = predicted_move.to("cuda")

        self.optimizer.zero_grad()
        loss = self.criterion(target, predicted_move)
        loss.backward()
        self.optimizer.step()

