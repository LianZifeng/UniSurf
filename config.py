import argparse


def load_config():
    # args
    parser = argparse.ArgumentParser(description="UniSurf")

    # data
    parser.add_argument('--data_path', default=r"", type=str, help="path to data")
    parser.add_argument('--excel_path', default=r"", type=str, help="path to data list")
    parser.add_argument('--surf_hemi', default="left", type=str, help="left or right hemisphere")
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
    # training
    parser.add_argument('--base_lr', default=1e-4, type=float, help="base learning rate")
    parser.add_argument('--n_epochs', default=100, type=int, help="total training epochs")
    parser.add_argument('--valid_interval', default=10, type=int, help="validate model after each n epoch")
    parser.add_argument('--output_dir', default=r'', type=str, help='directory to save models')
    parser.add_argument("--resume", default=True, type=bool, help="resume training from pretrained checkpoint")
    # pial model
    parser.add_argument('--nc', default=256, type=int, help="num of channels")
    parser.add_argument('--K', default=7, type=int, help="kernal size")
    parser.add_argument('--n_scale', default=5, type=int, help="num of scales for image pyramid")
    parser.add_argument('--n_smooth', default=1, type=int, help="num of Laplacian smoothing layers")
    parser.add_argument('--lambd', default=1.0, type=float, help="Laplacian smoothing weights")

    config = parser.parse_args()

    return config