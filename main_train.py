import os

from loguru import logger

import SinGAN.functions as functions
from SinGAN.manipulate import SinGAN_generate
from SinGAN.training import train
from config import get_arguments


def cli():
    parser = get_arguments()
    parser.add_argument("--input_dir", help="input image dir", default="Input/Images")
    parser.add_argument("--input_name", help="input image name", required=True)
    parser.add_argument("--mode", help="task to be done", default="train")
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    main(opt)


def main(opt):
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if os.path.exists(dir2save):
        logger.info("Trained model directory already exists")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)


if __name__ == "__main__":
    cli()
