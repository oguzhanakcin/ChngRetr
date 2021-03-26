import argparse
from utils.dataload import get_img_locs,create_test_trainset




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1img-dir",type=str,default="./../dataset20k/change/t1/")
    parser.add_argument("--out-loc",type=str,default="./")
    parser.add_argument("--train-ratio",type=float,default=0.8)
    opt = parser.parse_args()

    t1_locs = get_img_locs(opt.t1img_dir)
    create_test_trainset(t1_locs,opt.train_ratio,opt.out_loc)