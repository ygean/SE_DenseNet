from pathlib import Path
import torch

from tqdm import tqdm


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=5):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        for data, target in tqdm(data_loader, ncols=80):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data.item() / len(data_loader))
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print()
        mode = "train" if is_train else "test"
        print(">>>[{}] loss: {:.4f}/accuracy: {:.4f}".format(mode, sum(loop_loss), sum(accuracy) / len(data_loader.dataset) ))
        return mode, sum(loop_loss), sum(accuracy) / len(data_loader.dataset)

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            mode, loss, correct = self._iteration(data_loader)
            return mode, loss, correct

    def test(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            mode, loss, correct = self._iteration(data_loader, is_train=False)
            return mode, loss, correct

    def loop(self, epochs, train_data, test_data, scheduler=None):
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            # save statistics into txt file
            self.save_statistic(*((ep,) + self.train(train_data)))
            self.save_statistic(*((ep,) + self.test(test_data)))
            if ep % self.save_freq:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch, "net_state_dict": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / "model_epoch_{}.ckpt".format(epoch))
    
    def save_statistic(self, epoch, mode, loss, accuracy):
        with open("state.txt", "a", encoding="utf-8") as f:
            f.write(str({"epoch": epoch, "mode": mode, "loss": loss, "accuracy": accuracy}))
            f.write("\n")
