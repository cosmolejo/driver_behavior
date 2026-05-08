import hydra
import torch
from omegaconf import DictConfig
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from models.setup_factory import SetupFactory

@hydra.main(version_base=None, config_path="configs", config_name="config_local")
def main(cfg: DictConfig):

    model = SetupFactory.get_model(cfg.setup.mode)(cfg)
    model.load_state_dict(torch.load('pretrained_weights/'+cfg.checkpoint_file+'.pth.tar', weights_only=True)['model_state_dict'])
    model = model.eval()
    example_inputs = (torch.rand(1, 16, 3, 224, 224),)
    # torch.Size([1, 16, 3, 224, 224])

    exported_program = torch.export.export(model, example_inputs)

    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    # Save to .pte file
    with open("model.pte", "wb") as f:
        f.write(program.buffer)


if __name__ == '__main__':
    main()
