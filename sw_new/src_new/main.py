from pathlib import Path
from trainer import train_pipeline

if __name__ == "__main__":
    train_pipeline(
        data_dir=Path(__file__).parent.parent.resolve() / "dmd",   # Your data folder
        num_classes=3,
        sequence_length=32,  # Or -1 for full video
        sample_one_each=2,
        batch_size=2,
        num_epochs=500,
        lr=1e-4
    )