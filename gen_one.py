import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim

from model import Net

batch_size_train = 1000

random_seed = 1
torch.manual_seed(random_seed)

# We still need the dataset to generate image
# because we need to get the average to regularize it

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((28, 28)),
    ])), batch_size=batch_size_train, shuffle=True)

# Get the average
avg = torch.zeros((1, 1, 28, 28))
count = 0
for batch_idx, (data, t) in enumerate(train_loader):
    avg += data.mean(0)
    count += 1
avg = avg/count

# Fetch the trained model
network = Net()
network.load_state_dict(torch.load('./results/model_mnist.pth'))

# Get a batch of empty canvases
bsize = 1
sample = torch.zeros((bsize, 1, 28, 28), requires_grad=True)

# Our target class
target = torch.tensor([5 for _ in range(bsize)])

maxiter = 100
alpha = 0.05
beta = 0.
eps = 1e-5
lamb = 50
losses = []

sample_opt = optim.SGD([sample], lr=alpha)
for i in range(maxiter):
    # Mix the sample with noise
    mixed = sample + beta*torch.randn_like(sample)
    # Get the current prediction
    pred = network(mixed)
    # Calculate loss with regularization
    loss = F.nll_loss(pred, target) + lamb*F.mse_loss(sample, avg)
    losses.append(loss.item())

    # One step forward
    sample_opt.zero_grad()
    loss.backward()
    sample_opt.step()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set(title="Average of All Train Images", xticks=[], yticks=[])
ax1.imshow(avg.detach().numpy()[0][0], cmap='gray')

ax2.set(title="Generated Image of Class %d" % (target), xticks=[], yticks=[])
ax2.imshow(torch.mean(sample, axis=0).detach().numpy()[0], cmap='gray')

ax3.set(title="Loss Over Epochs")
ax3.plot(losses)

plt.show()
