import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

from ixai.data import ImageNetteDataset, ClickMetteDataset
from helpers import normalize_tensor, spearman_correlation, get_gen, seed_worker
from ixai.torch_explainers import SaliencyGraph


class ImageNette(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.net = resnet18()
        num_in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_in_features, 10)
        self.batch_size = 64

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        self.loss = nn.CrossEntropyLoss()

        warm_up = 5
        warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1, total_iters=warm_up)
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=90, eta_min=0)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer,
                                                               schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                                                               milestones=[warm_up])

        self.debug = debug

    def forward(self, x):
        out = self.net(x)
        return out

    def fit(self, exp_loss, n_epochs=90, weight=1):
        dataset = ClickMetteDataset(remove_dup=True)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, worker_init_fn=seed_worker,
                            generator=get_gen())
        self.val_loader = DataLoader(ImageNetteDataset(set='val'), batch_size=self.batch_size,
                                     worker_init_fn=seed_worker,
                                     generator=get_gen())

        for epoch in range(0, n_epochs):
            total_pred_loss = 0
            total_exp_loss = 0
            for batch in tqdm(loader, 'Training'):
                self.train()
                self.optimizer.zero_grad()

                batch_X, batch_y, batch_ix = batch

                l_pred = self.loss(self(batch_X.cuda()).squeeze(), batch_y.cuda())
                l_exp = exp_loss(self, [batch_X.cuda(), batch_y.cuda()], dataset.get_exp(batch_ix).cuda())

                l = l_exp + weight * l_pred
                l.backward()
                self.optimizer.step()
                total_pred_loss += l_pred.item() * len(batch_X)
                total_exp_loss += l_exp.item() * len(batch_X)
            total_pred_loss /= len(dataset)
            total_exp_loss /= len(dataset)

            self.scheduler.step()

            self.eval()

            total_val_loss, acc1 = self.validate()

            # Compute similarity and MSE
            sim, mse = self.validate_exps()

            print(
                f'{epoch + 1}/{n_epochs} | Train loss:{total_pred_loss:.3g} (pred) + {total_exp_loss:.3g} (exp) | Val loss:{total_val_loss:.3g} | Val acc:{acc1} | Val exp similarity:{sim}')

    def validate(self, dataset=None):
        loader = self.val_loader
        self.eval()
        total_val_loss = 0
        total_sum_accs = 0
        for batch in tqdm(loader, 'Validation'):
            batch_X, batch_y = batch[0], batch[1]
            output = self(batch_X.cuda())
            val_l = self.loss(output.squeeze(), batch_y.cuda())
            total_val_loss += val_l.item() * len(batch_X)
            acc = accuracy(output, batch_y.cuda())
            total_sum_accs += acc * len(batch_X)
        total_val_loss /= len(loader.dataset)
        total_sum_accs /= len(loader.dataset)
        return total_val_loss, total_sum_accs

    def validate_exps(self, explainer=None, normalize='gaussian', dataset=None):
        dataset = ClickMetteDataset(set='val', remove_dup=True) if dataset is None else dataset
        loader = DataLoader(dataset, batch_size=self.batch_size, worker_init_fn=seed_worker, generator=get_gen())
        self.eval()
        total_sim = 0
        total_mse = 0
        for batch in tqdm(loader, 'Explanations'):
            batch_X, batch_y, batch_ix = batch
            explainer = SaliencyGraph(self.forward) if explainer is None else explainer
            attr = explainer.attribute(batch_X.cuda(), target=batch_y.cuda(), abs=True)
            sim = spearman_correlation(torch.mean(attr, dim=1), dataset.get_exp(batch_ix))
            mse = torch.nn.MSELoss()(normalize_tensor(torch.mean(attr, dim=1), method=normalize, samplewise=True),
                                     normalize_tensor(dataset.get_exp(batch_ix).cuda(), method=normalize,
                                                      samplewise=True))

            total_sim += sim * len(batch_X)
            total_mse += mse * len(batch_X)
        total_sim /= len(loader.dataset)
        total_mse /= len(loader.dataset)
        return total_sim, total_mse


def accuracy(pred, test):
    pred = torch.argmax(pred, dim=1)
    return (pred == test).float().mean().item()
