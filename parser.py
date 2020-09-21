import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="Batch size"
)

parser.add_argument(
    "--workers",
    default=16,
    type=int,
    help="Workers number"
)

parser.add_argument(
    "--cuda",
    default=0,
    type=int,
    help="Cuda device"
)

parser.add_argument(
    "--checkpoints",
    default="",
    type=str,
    help="Path to a folder with 'checkpoints' folder"
)

parser.add_argument(
    "--model",
    default="resnet18",
    type=str,
    help="Model name",
    choices=["resnet50", "resnet18", "resnet50lcm", "resnet18lcm", "se_resnet18"]
)

parser.add_argument(
    "--dataset_id",
    default=None,
    type=int,
    help="Dataset id to process"
)

args = parser.parse_args()

