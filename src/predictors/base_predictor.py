# src/inference.py
import joblib
# 1. IMPORTS
# Importar la arquitectura desde src.models
# Importar utilidades de preprocesamiento desde src.utils
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class Predictor:
    def __init__(self, model, weights, config):
        """
        Fase de Preparación: Solo se ejecuta una vez.
        """
        self.cfg = config
        self.model = model
        self.load_weights(weights)
        self.model.eval()  # Poner en modo evaluación
        self.le = joblib.load('../datasets/label_encoder.joblib')
        # set cuda flag
        self.is_cuda = torch.cuda.is_available()

        self.cuda = self.is_cuda & self.cfg.cuda

        # set the manual seed for torch
        self.manual_seed = self.cfg.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.cfg.gpu_device)
            self.model = self.model.to(self.device)

    def preprocess(self, raw_data):
        """
        Transforma datos externos (ej. una imagen en disco o un JSON)
        al formato tensor que el modelo espera.
        """
        # 1. Convertir a tensor y mover canales: (n, H, W, C) -> (n, C, H, W)
        # Usamos float32 directamente para evitar conversiones posteriores
        # tensors = torch.from_numpy(raw_data).permute(0, 3, 1, 2).float()
        #
        # # 2. Normalizar a (Equivalente a ToTensor() / 255)
        # tensors /= 255.0
        #
        # # 3. Resize redimensionando todo el lote a la vez
        # # El antialias=True es importante para mantener la calidad visual en el downsampling
        # tensors = F.interpolate(
        #     tensors,
        #     size=(224, 224),
        #     mode='bilinear',
        #     align_corners=False,
        #     antialias=True
        # )
        tensors = [transforms.ToTensor()(
            Image.fromarray(np.uint8(frame * 255)).resize((224, 224))
        ) for frame in raw_data]

        tensors = torch.stack(tensors)
        return tensors

    def predict(self, raw_data_package):
        """
        Flujo principal de inferencia.
        """
        input_tensor = self.preprocess(raw_data_package)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)

        return self.postprocess(output)

    def postprocess(self, output):
        pred = torch.argmax(output, dim=1)
        pred_numpy = pred.cpu().numpy()
        return self.le.inverse_transform(pred_numpy)
    def load_weights(self, file_name) -> None:
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """

        checkpoint = torch.load('pretrained_weights/' + file_name + '.pth.tar')
        self.model.load_state_dict(checkpoint['model_state_dict'])

# 2. PUNTO DE ENTRADA (Main)
if __name__ == "__main__":
    # Configurar rutas
    WEIGHTS = "pretrained_weights/best_model.pth"
    DATA_EXTERNO = "ruta/a/tus/datos/externos"

    # Instanciar predictor
    predictor = Predictor(model_path=WEIGHTS, config_path="src/configs/deploy.yaml")

    # Ejecutar
    resultado = predictor.predict(DATA_EXTERNO)
    print(f"Resultado de la IA: {resultado}")