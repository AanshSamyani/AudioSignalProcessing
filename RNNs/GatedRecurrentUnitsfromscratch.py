import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example


class GRUfromscratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUfromscratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.softmax_layer = nn.LogSoftmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate_hidden_state = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        r_t = self.reset_gate(combined)
        r_t = self.sigmoid(r_t)
        
        resetted_memory = r_t * hidden_tensor
        combined_2 = torch.cat((resetted_memory, input_tensor), 1)
        
        h_tilde_t = self.candidate_hidden_state(combined_2)
        h_tilde_t = self.tanh(h_tilde_t)
        
        z_t = self.update_gate(combined)
        z_t = self.sigmoid(z_t)
        
        next_hidden_state = (1-z_t) * hidden_tensor + z_t * h_tilde_t
        output = self.output_layer(next_hidden_state)
        output = self.softmax_layer(output)
        return output, next_hidden_state
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
category_lines, all_categories = load_data()
n_categories = len(all_categories)

n_hidden = 128 ## Hyperparemeter - Can be tuned
gru = GRUfromscratch(N_LETTERS, n_hidden, n_categories)

# one step
input_tensor = letter_to_tensor('A')
hidden_tensor = gru.init_hidden()

output, next_hidden = gru(input_tensor, hidden_tensor)
print(output.size())
print(next_hidden.size())

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

## (NLLLoss) -> The negative log likelihood loss. 
## It is useful to train a classification problem with C classes.
criterion = nn.NLLLoss()   
learning_rate = 0.005
optimizer = torch.optim.SGD(gru.parameters(), lr=learning_rate)

def train(line_tensor, category_tensor):
    hidden_tensor = gru.init_hidden()
    
    for i in range(len(line_tensor)):
        output, hidden_tensor = gru(line_tensor[i], hidden_tensor)
        
    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
    
    output, loss = train(line_tensor, category_tensor)
    current_loss += loss 
    
    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0
        
    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")        
        
        
plt.figure()
plt.plot(all_losses)
plt.show()    
