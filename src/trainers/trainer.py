import logging

import hydra
import mlflow
import numpy as np
import torch
from tqdm import tqdm
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from utils.misc import print_cuda_statistics
from .base import BaseTrainer

cudnn.benchmark = True


def _confusion_matrix_figure(cm: np.ndarray, class_names: list, normalize: bool = True):
    """
    Crea una figura matplotlib de la matriz de confusión, lista para
    writer.add_figure de TensorBoard.

    Si normalize=True, normaliza por fila (true labels) -> cada fila suma 1.
    Más informativo con clases desbalanceadas: lees "de los samples reales
    de clase X, qué porcentaje se predijo como cada clase".
    """
    import matplotlib
    matplotlib.use('Agg')  # backend sin display, seguro en headless
    import matplotlib.pyplot as plt

    if normalize:
        with np.errstate(all='ignore'):
            cm_display = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
            cm_display = np.nan_to_num(cm_display)
        fmt = '.2f'
        title = 'Confusion Matrix (normalized by row)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    thresh = cm_display.max() / 2.0 if cm_display.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm_display[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm_display[i, j] > thresh else 'black')

    fig.tight_layout()
    return fig


class SafeDrivingTrainer(BaseTrainer):

    def __init__(self, config, data_module=None, model=None, loss=None, optimizer=None) -> None:
        self.config = config
        self.logger = logging.getLogger(self.config.setup.trainer)

        self.test_loss = 100000
        self.best_loss = 100000
        self.train_loss = None

        self.model = model
        self.data_module = data_module
        self.data_module.setup()

        self.train_loader = self.data_module.train_dataloader()
        self.test_loader = self.data_module.test_dataloader()
        self.val_loader = self.data_module.val_dataloader()

        self.loss = loss
        self.optimizer = optimizer
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # Nombres de clase para reportes; intenta inferir del LabelEncoder
        self.class_names = self._infer_class_names()

        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

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

        self.load_checkpoint(self.config.checkpoint_file)
        self.writer = SummaryWriter(
            log_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )

    def _infer_class_names(self):
        """
        Extrae nombres de clase del LabelEncoder del dataset, si está disponible.
        Soporta tanto dataset directo como envuelto en Subset (tras random_split).
        Si falla, devuelve etiquetas genéricas class_0, class_1, ...
        """
        candidates = [
            lambda: self.data_module.train_dataloader().dataset.dataset.le.classes_,
            lambda: self.data_module.train_dataloader().dataset.le.classes_,
        ]
        for getter in candidates:
            try:
                return list(getter())
            except (AttributeError, TypeError):
                continue
        num = getattr(self.config.model, 'num_classes', 3)
        return [f"class_{i}" for i in range(num)]

    def load_checkpoint(self, file_name) -> None:
        try:
            checkpoint = torch.load('pretrained_weights/' + file_name + '.pth.tar',
                                    weights_only=False)
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = checkpoint['optimizer']
        except FileNotFoundError:
            self.logger.info("Checkpoint file not found. Starting from scratch")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer,
        }
        if is_best:
            torch.save(checkpoint, 'pretrained_weights/' + file_name + '.best.pth.tar')
        torch.save(checkpoint, 'pretrained_weights/' + file_name + '.pth.tar')

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
            self.finalize()

    def train(self):
        with mlflow.start_run():
            mlflow.log_param("Config", self.config)
            for epoch in range(self.current_epoch, self.config.max_epoch + 1):
                self.train_one_epoch(epoch)
                previous_loss = self.test_loss
                self.test(epoch)
                if self.test_loss < self.best_loss:
                    self.save_checkpoint(file_name=self.config.checkpoint_file, is_best=1)
                    self.best_loss = self.test_loss
                else:
                    self.save_checkpoint(file_name=self.config.checkpoint_file, is_best=0)
                if (previous_loss is not None
                        and abs(previous_loss - self.test_loss) < self.config.early_stopping_delta):
                    self.logger.info(f"Early stopping!! Delta: {previous_loss - self.test_loss}")
                    break
                self.current_epoch += 1

    def train_one_epoch(self, epoch):
        total_loss = 0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            self.writer.add_scalar("Loss/train_batch", loss, batch_idx)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f'Train Epoch: {self.current_epoch} '
                    f'[{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}'
                )
            mlflow.log_metric("loss", loss.item(), step=self.current_iteration)
            self.current_iteration += 1

        avg_epoch_loss = total_loss / len(self.train_loader.dataset)
        self.train_loss = avg_epoch_loss
        self.writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
        self.writer.add_scalar("Total_Loss/train", total_loss, epoch)

    def _evaluate(self, loader, split_name: str):
        """
        Lógica común de evaluación: corre el modelo y devuelve métricas +
        arrays de predicciones/targets. Privada; usada por test() y validate().
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(loader, desc=f"Eval {split_name}"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.loss(output, target).item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        n_samples = len(loader.dataset)

        loss = total_loss / n_samples
        accuracy = 100.0 * correct / n_samples
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(
            all_targets, all_preds,
            labels=list(range(len(self.class_names))),
        )

        return {
            'loss': loss, 'accuracy': accuracy, 'correct': correct, 'n_samples': n_samples,
            'precision': precision, 'recall': recall, 'f1': f1,
            'confusion_matrix': cm,
            'all_preds': all_preds, 'all_targets': all_targets,
        }

    def _log_evaluation(self, metrics: dict, split_name: str):
        """Loguea métricas + matriz de confusión + métricas por clase a TB y consola."""
        step = self.current_epoch
        loss = metrics['loss']
        accuracy = metrics['accuracy']
        cm = metrics['confusion_matrix']
        all_preds = metrics['all_preds']
        all_targets = metrics['all_targets']

        # Escalares globales (mantengo Metrics/* sin sufijo para compatibilidad
        # con runs anteriores donde el "test" era la evaluación canónica)
        self.writer.add_scalar(f"Loss/{split_name}", loss, step)
        self.writer.add_scalar("Metrics/Accuracy", accuracy, step)
        self.writer.add_scalar("Metrics/Precision", metrics['precision'], step)
        self.writer.add_scalar("Metrics/Recall", metrics['recall'], step)
        self.writer.add_scalar("Metrics/F1-Score", metrics['f1'], step)

        # Matrices de confusión: normalizada (lectura por porcentajes) y bruta (conteos)
        fig_norm = _confusion_matrix_figure(cm, self.class_names, normalize=True)
        self.writer.add_figure(f"ConfusionMatrix/{split_name}_normalized", fig_norm, step)
        fig_raw = _confusion_matrix_figure(cm, self.class_names, normalize=False)
        self.writer.add_figure(f"ConfusionMatrix/{split_name}_counts", fig_raw, step)

        # Métricas por clase: críticas para detectar colapso a la clase mayoritaria
        labels = list(range(len(self.class_names)))
        per_class_p = precision_score(all_targets, all_preds, average=None,
                                      zero_division=0, labels=labels)
        per_class_r = recall_score(all_targets, all_preds, average=None,
                                   zero_division=0, labels=labels)
        per_class_f1 = f1_score(all_targets, all_preds, average=None,
                                zero_division=0, labels=labels)
        for i, cname in enumerate(self.class_names):
            self.writer.add_scalar(f"PerClass_{split_name}/precision_{cname}",
                                   per_class_p[i], step)
            self.writer.add_scalar(f"PerClass_{split_name}/recall_{cname}",
                                   per_class_r[i], step)
            self.writer.add_scalar(f"PerClass_{split_name}/f1_{cname}",
                                   per_class_f1[i], step)

        # Consola
        self.logger.info(
            f"\n{split_name.capitalize()} set: Average loss: {loss:.4f}, "
            f"Accuracy: {metrics['correct']}/{metrics['n_samples']} ({accuracy:.0f}%), "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}\n"
        )
        report = classification_report(
            all_targets, all_preds, labels=labels,
            target_names=self.class_names, zero_division=0,
        )
        self.logger.info(f"\nClassification report ({split_name}):\n{report}")
        self.logger.info(f"Confusion matrix ({split_name}):\n{cm}\n")

    def test(self, epoch=None):
        """Evaluación cíclica usada para selección de checkpoint y early stopping."""
        metrics = self._evaluate(self.test_loader, split_name='test')
        self.test_loss = metrics['loss']
        if epoch is not None:
            self.writer.add_scalars(
                "Experiment_Loss",  # <- mismo plot
                {
                    "train": self.train_loss,
                    "test": self.test_loss,
                },
                epoch
            )
        self._log_evaluation(metrics, split_name='test')

    def validate(self):
        """
        Evaluación en val_loader (held-out final reservado para demos y
        comparación de modelos). Llamar manualmente cuando se quiera evaluar
        un modelo entrenado contra este conjunto que no ha visto durante el
        entrenamiento ni la selección de checkpoint.
        """
        metrics = self._evaluate(self.val_loader, split_name='val')
        self._log_evaluation(metrics, split_name='val')
        return metrics

    def finalize(self):
        self.writer.close()