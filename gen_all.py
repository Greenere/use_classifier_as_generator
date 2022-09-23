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


def get_sample_mle(sample, target, maxiter, alpha, beta, lamb, pgd=False):
    """
    Get sample through maximum likelihood estimation:
    sample: sample canvas
    target: target class
    maxiter: maximum iterations
    alpha: step size
    beta: mixture strength
    lamb: regularization strength
    pgd: use PGD or not
    """
    losses = []
    sample_opt = optim.SGD([sample], lr=alpha)
    for i in range(maxiter):
        mixed = sample + beta*torch.randn_like(sample)
        pred = network(mixed)
        loss = F.nll_loss(pred, target) + lamb*F.mse_loss(mixed, avg)
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

samples = []
losses = []
losses2 = []
preds = []
for i in range(10):
    bsize = 1
    sample = torch.zeros((bsize, 1, 28, 28))
    sample = torch.tensor(sample.numpy(), requires_grad=True)
    target = torch.tensor([i for _ in range(bsize)])
    sample, losses = get_sample_mle(sample, target,
                                    maxiter, alpha, beta, lamb, pgd=True)
    preds.append(network(sample).argmax())
    samples.append(sample)
    losses2.append(losses)


fig, axes = plt.subplots(2, 10)
for i, ax in enumerate(axes[0]):
    # Generating 0 and predicted as 0 will be titled: G 0 | P 0
    ax.set(title="G %d | P %d" % (i, preds[i]))
    ax.set(xticks=[], yticks=[])
    ax.imshow(samples[i].detach().numpy()[0][0], cmap='gray')

for i, ax in enumerate(axes[1]):
    ax.set(xticks=[], yticks=[])
    ax.plot(losses2[i])

plt.tight_layout()
plt.show()
