import os
import sys
import argparse

parser = argparse.ArgumentParser()
# Path setting
parser.add_argument(
    "--model_dir",
    type=str,
    default='.model/',
    help="The path to the pretrained Mask2former to use for mask generation cached locally.",
)
parser.add_argument("--data_path", 
                    type=str,
                    default="/data/AVSBench-semantic/", 
                    help="The path to the datasets.")
parser.add_argument("--feature_path", 
                    type=str,
                    default="/data/AVSBench-semantic/", 
                    help="The path to the processed audio feature.")

#Training setting
parser.add_argument("--gpu_id", type=str, default="0", help="The GPU device to run generation on.")
parser.add_argument("--bs",  type=int, default=2, help="batch_size for training")
parser.add_argument("--num_workers",  type=int, default=10, help="the number of workers for training")
parser.add_argument("--lr", type=float, default=1e-4, help='lr to fine tuning adapters.')
parser.add_argument("--weight_dec", type=float, default=0.05, help='weight decay to fine tuning adapters.')
parser.add_argument("--epochs", type=int, default=30, help='epochs to fine tuning adapters.')
parser.add_argument("--device", type=str, default="cuda:0", help="The device to run generation on.")
parser.add_argument("--log_dir", type=str, default="./log", help="The path to save checkpoint and mask.")
parser.add_argument("--mask_path", 
                    type=str,
                    default="/data/avs_mask_final/", 
                    help="The path to the first stage results of binary mask.")

# Testing setting
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default='./log',
    help="The path to the checkpoint to for testing.",
)
parser.add_argument("--save_mask", default=False, action='store_true', help="whether save the test set mask.")

args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id