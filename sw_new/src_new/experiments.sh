conda activate tesis

python main.py loss_fn=cross_entropy
python main.py loss_fn=CE_label_smoothing loss_kwargs="{ 'label_smoothing': 0.1}"
python main.py loss_fn=CE_label_smoothing loss_kwargs="{ 'label_smoothing': 0.25}"
python main.py loss_fn=CE_label_smoothing loss_kwargs="{ 'label_smoothing': 0.5}"
python main.py loss_fn=CE_label_smoothing loss_kwargs="{ 'label_smoothing': 0.75}"
python main.py loss_fn=sigmoid_focal_loss
#python main.py loss_fn=sigmoid_focal_loss loss_kwargs="{ 'reduction':'mean'}"
#python main.py loss_fn=sigmoid_focal_loss loss_kwargs="{ 'alpha': 0.1}"
#python main.py loss_fn=sigmoid_focal_loss loss_kwargs="{ 'alpha': 0.1, 'reduction':'mean'}"
#python main.py loss_fn=sigmoid_focal_loss loss_kwargs="{ 'alpha': 0.5}"
#python main.py loss_fn=sigmoid_focal_loss loss_kwargs="{ 'alpha': 0.5, 'reduction':'mean'}"
#python main.py loss_fn=sigmoid_focal_loss loss_kwargs="{ 'alpha': 0.75}"
#python main.py loss_fn=sigmoid_focal_loss loss_kwargs="{ 'alpha': 0.75, 'reduction':'mean'}"


