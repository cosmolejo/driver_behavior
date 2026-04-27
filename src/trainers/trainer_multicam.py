"""
Mnist Main agent, as mentioned in the tutorial
"""
import mlflow
import torch
from tqdm import tqdm
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from utils.misc import print_cuda_statistics
from .base import BaseTrainer
from sklearn.metrics import precision_score, recall_score, f1_score
import hydra
cudnn.benchmark = True

import logging


class MultiCamTrainer(BaseTrainer):

    def __init__(self, config, data_module=None, model=None, loss=None, optimizer=None) -> None:


        self.test_loss = None
        self.best_loss = None

        self.config = config
        self.logger = logging.getLogger(self.config.setup.trainer)

        # define models
        self.model = model


        self.data_module = data_module

        self.data_module.setup()

        # define data_loader
        self.train_loader = self.data_module.train_dataloader()
        self.test_loader = self.data_module.test_dataloader()





        # define loss
        self.loss = loss

        # define optimizer
        self.optimizer = optimizer
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.writer = SummaryWriter(log_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    def load_checkpoint(self, file_name) -> None:
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:
            checkpoint = torch.load('pretrained_weights/' + file_name+'.pth.tar')
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = checkpoint['optimizer']
        except FileNotFoundError:
            self.logger.info("Checkpoint file not found. Starting from scratch")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        # Save the state
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer,

        }
        if is_best:
            torch.save(checkpoint, 'pretrained_weights/' + file_name+'.best.pth.tar')
        torch.save(checkpoint, 'pretrained_weights/' + file_name + '.pth.tar')
    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        with mlflow.start_run():
            mlflow.log_param("Config", self.config)
            for epoch in range(self.current_epoch, self.config.max_epoch + 1):
                total_loss = 0

                self.model.train()
                for batch_idx, (data_face,data_body, target) in enumerate(self.train_loader):
                    data_face, data_body,target_face = data_face.to(self.device),data_body.to(self.device), target_face.to(self.device)


                    self.optimizer.zero_grad()
                    output = self.model(data_face,data_body)
                    loss = self.loss(output, target)
                    self.writer.add_scalar("Loss/train_batch", loss, batch_idx)

                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()* data_face.size(0)

                    if batch_idx % self.config.log_interval == 0:
                        self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            self.current_epoch, batch_idx * len(data_face), len(self.train_loader.dataset),
                                                100. * batch_idx / len(self.train_loader), loss.item()))



                    mlflow.log_metric("loss", loss.item(), step=self.current_iteration)
                    self.current_iteration += 1
            avg_epoch_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
            self.writer.add_scalar("Total_Loss/train", loss, epoch)
            self.test()
            if self.test_loss < self.best_loss or self.best_loss is None:
                self.save_checkpoint(file_name=self.config.checkpoint_file, is_best=1)
                self.best_loss = self.test_loss
            else:
                self.save_checkpoint(file_name=self.config.checkpoint_file, is_best=0)

            self.current_epoch += 1
    def train_one_epoch(self,epoch):
        """
        One epoch of training
        :return:
        """
        pass



    def test(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
                i = 0
                for data_face,data_body, target, target in tqdm(self.test_loader):
                    data_face, data_body,target_face = data_face.to(self.device),data_body.to(self.device), target_face.to(self.device)
                    output = self.model(data_face,data_body)
                    test_loss += self.loss(output, target ).item()* data_face.size(0)  # sum up batch loss
                    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    all_preds.extend(pred.view(-1).cpu().numpy())
                    all_targets.extend(target.view(-1).cpu().numpy())

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        self.test_loss = test_loss
        self.writer.add_scalar("Loss/test", test_loss, self.current_epoch)
        self.writer.add_scalar("Metrics/Accuracy", accuracy, self.current_epoch)
        self.writer.add_scalar("Metrics/Precision", precision, self.current_epoch)
        self.writer.add_scalar("Metrics/Recall", recall, self.current_epoch)
        self.writer.add_scalar("Metrics/F1-Score", f1, self.current_epoch)
        self.logger.info(
            f'\nTest set: Average loss: {test_loss:.4f}, '
            f'Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.0f}%), '
            f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n'
        )
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.writer.close()

