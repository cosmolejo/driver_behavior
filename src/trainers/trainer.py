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

cudnn.benchmark = True

import logging


class SafeDrivingTrainer(BaseTrainer):

    def __init__(self, config, data_module=None, model=None, loss=None, optimizer=None) -> None:
        self.config = config
        self.logger = logging.getLogger(self.config.trainer)


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
        self.writer = SummaryWriter()




    def load_checkpoint(self, file_name) -> None:
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:
            checkpoint = torch.load('checkpoint.pth')

            self.model.load_state_dict(torch.load( checkpoint['model_state_dict'], weights_only=True))
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            'epoch': 10,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),

        }
        torch.save(checkpoint, 'pretrained_weights/'+ file_name)


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
        for epoch in range(1, self.config.max_epoch + 1):
            total_loss = 0
            with mlflow.start_run():
                mlflow.log_param("Config", self.config)
                self.model.train()
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.loss(output, target)
                    self.writer.add_scalar("Loss/train_batch", loss, batch_idx)

                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                    if batch_idx % self.config.log_interval == 0:
                        self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            self.current_epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                                100. * batch_idx / len(self.train_loader), loss.item()))



                    mlflow.log_metric("loss", loss.item(), step=self.current_iteration)
                    self.current_iteration += 1
            avg_epoch_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
            self.writer.add_scalar("Total_Loss/train", loss, epoch)
            self.validate()
            self.save_checkpoint()

            self.current_epoch += 1
    def train_one_epoch(self,epoch):
        """
        One epoch of training
        :return:
        """
        pass



    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
                i = 0
                for data, target in tqdm(self.test_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += self.loss(output, target ).item()  # sum up batch loss
                    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    # bar.update(100 *i / len(self.test_loader))
                    # i += 1

        test_loss /= len(self.test_loader.dataset)
        self.writer.add_scalar("Loss/test", test_loss, self.current_epoch)
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.writer.close()

