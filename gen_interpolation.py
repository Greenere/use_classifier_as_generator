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

def get_sample_mle_multi(sample, target1, target2, weight1, weight2, maxiter, alpha, beta, lamb, pgd=False):
    losses = []
    sample_opt = optim.SGD([sample], lr=alpha)
    for i in range(maxiter):
        mixed = sample + beta*torch.randn_like(sample)
        pred = network(mixed)
        loss = -pred[:, target1] * weight1 - pred[:, target2] * weight2 + \
            lamb*F.mse_loss(mixed, avg)
        losses.append(loss.item())

        sample_opt.zero_grad()
        loss.backward()

        if pgd:
            grads = sample.grad.detach()

            sample = torch.tensor(
                sample.detach().numpy() - alpha*grads.numpy())
            sample = torch.clip(sample, 0, 1)
            sample = torch.tensor(sample.detach().numpy(), requires_grad=True)
        else:
            sample_opt.step()
    return sample, losses

maxiter = 100
alpha = 0.05
beta = 0.
lamb = 50
weight1 = 0.5
weight2 = 0.5

samples = []
for i in range(10):
    for j in range(10):
        bsize = 1
        sample = torch.zeros((bsize, 1, 28, 28))
        sample = torch.tensor(sample.numpy(), requires_grad=True)
        target1 = torch.tensor([i for _ in range(bsize)])
        target2 = torch.tensor([j for _ in range(bsize)])
        sample, losses = get_sample_mle_multi(
            sample, target1, target2, weight1, weight2,\
            maxiter, alpha, beta, lamb)
        samples.append(sample)

plt.figure(figsize=(20, 20))
for i in range(100):
    plt.tight_layout()
    plt.subplot(10, 10, i+1)
    plt.imshow(samples[i].detach().numpy()[0][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.savefig("./results/generated_interpolations.png")
