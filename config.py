import os
import torch

class Config:
    # paths (relative to project root, portable across OS)
    _project_root = os.path.dirname(os.path.abspath(__file__))

    # dataset (override with --data_root on different machines)
    data_root = r"/home/hipeson/WorkSpace/DZ/datasets/nyu_labeled"
    scale = 8
    train_split = 1000
    crop_size = 256  # divisible by 8 (LCM of patch_size=4 and window_size=8)
    repeat = 32       # each image sampled repeat times per epoch (A2GS style)

    # training
    batch_size = 8
    num_epochs = 300
    lr = 5e-4
    lr_gamma = 0.5           # ReduceLROnPlateau: decay factor
    lr_patience = 10          # ReduceLROnPlateau: patience epochs
    weight_decay = 1e-4
    grad_clip = 0             # gradient clipping max norm (0 = disabled)
    num_workers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Swin encoder
    swin_embed_dim = 64
    swin_depths = [2, 2, 2, 2]       # num blocks per stage (A2GS style)
    swin_num_heads = [2, 2, 2, 2]     # heads per stage (A2GS style, all=2)
    swin_window_size = 8
    swin_mlp_ratio = 2.0
    swin_drop_path_rate = 0.1         # stochastic depth

    # fusion
    fusion_num_heads = 2

    # loss weights
    loss_l1_weight = 1.0
    loss_edge_weight = 0.1

    # version tag (saved in checkpoints, printed in logs)
    model_version = "5.7.2"

    # checkpoint (portable: relative to project root)
    checkpoint_dir = os.path.join(_project_root, "checkpoints")
    log_dir = os.path.join(_project_root, "logs")

cfg = Config()
