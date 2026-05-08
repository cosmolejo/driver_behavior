import numpy as np
import cv2 as cv
import hydra
from omegaconf import DictConfig
import joblib
from data.dmd import DMD
from models.setup_factory import SetupFactory
from predictors.base_predictor import Predictor
from tqdm import tqdm
import time
import torch
from executorch.runtime import Runtime
from typing import List

def get_label_encoder(cfg):
    full_dataset = DMD(cfg)
    joblib.dump(full_dataset.le, '../datasets/label_encoder.joblib')

    # Opcional: Verificar qué clases memorizó
    print("Clases guardadas:", full_dataset.le.classes_)


def run_video(cap,fps,ancho, alto, predictor):
     # 2. Configurar el VideoWriter (usamos mp4v para compatibilidad)
    ruta_salida = 'demo_salida.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))

    frame_buffer = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
        label = 'loading buffer'

        if len(frame_buffer) > 15:
            label = predictor.predict(np.array(frame_buffer))
            frame_buffer.pop(0)
        font = cv.FONT_HERSHEY_SIMPLEX

        # Use putText() method for
        # inserting text on video
        cv.putText(frame,
                    str(label),
                    (50, 50),
                    font, 1,
                    (0, 255, 255),
                    2,
                    cv.LINE_4)
        out.write(frame)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    
    out.release()
def run_latency_test(cap,num_samples, predictor):
    print("building samples")
    frame_buffer = []
    cont=1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
        label = 'loading buffer'

        if len(frame_buffer) > 15:
            break
    print("prediction estimation")
    
    start_time = time.time()
    with tqdm(total=num_samples) as pbar:
        for _ in range(num_samples):
            label = predictor.predict(np.array(frame_buffer))
            pbar.update(1)
    end_time = time.time()

    elapsed_time = end_time-start_time
    elapsed_time_avg = elapsed_time/num_samples

    return elapsed_time_avg


def executor_latency_test(cap, num_samples):
    runtime = Runtime.get()

    input_tensor: torch.Tensor = torch.randn(1, 16, 3, 224, 224)
    program = runtime.load_program("../models/model.pte")
    method = program.load_method("forward")



    start_time = time.time()
    with tqdm(total=num_samples) as pbar:
        for _ in range(num_samples):
            output: List[torch.Tensor] = method.execute([input_tensor])
            pbar.update(1)
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_time_avg = elapsed_time / num_samples

    return elapsed_time_avg


@hydra.main(version_base=None, config_path="configs", config_name="config_demo")
def main(cfg: DictConfig):

    #setting up the model
    model = SetupFactory.get_model(cfg.setup.mode)(cfg)

    predictor = Predictor(model,cfg.checkpoint_file, cfg)

    cap = cv.VideoCapture(f'{cfg.data_path}/{cfg.demo.video}')
    fps = cap.get(cv.CAP_PROP_FPS)
    ancho = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    match cfg.demo.mode:
        case "video":
            run_video(cap,fps,ancho,alto, predictor)
        case "latency":
            avg_lat = run_latency_test(cap,500, predictor)
            print(f'average latency: {avg_lat}')
        case "executor_latency":
            avg_lat = executor_latency_test(cap, 500)
            print(f'average latency: {avg_lat}')

    cap.release()

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()