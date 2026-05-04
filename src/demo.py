import numpy as np
import cv2 as cv
import hydra
from omegaconf import DictConfig
import joblib
from data.dmd import DMD
from models.setup_factory import SetupFactory
from predictors.base_predictor import Predictor

def get_label_encoder(cfg):
    full_dataset = DMD(cfg)
    joblib.dump(full_dataset.le, '../datasets/label_encoder.joblib')

    # Opcional: Verificar qué clases memorizó
    print("Clases guardadas:", full_dataset.le.classes_)
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    #setting up the model
    model = SetupFactory.get_model(cfg.setup.mode)(cfg)

    predictor = Predictor(model,cfg.checkpoint_file, cfg)

    cap = cv.VideoCapture('/home/antares/Tesis_Data/Vicomtech/dmd/gE/28/s2/gE_28_s2_2019-03-15T10_12_30+01_00_rgb_face_240.mp4')
    fps = cap.get(cv.CAP_PROP_FPS)
    ancho = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

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

    cap.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()