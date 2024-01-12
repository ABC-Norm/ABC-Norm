import argparse

parser = argparse.ArgumentParser(description='PyTorch Classification Tools')
parser.add_argument('--root', default='/home/v20180902/dataset/', metavar='DIR',
                    help='dataset path')
parser.add_argument('--task', default='cub', metavar='DIR',
                    help='cub, car, air')
parser.add_argument('--save_dir', default='checkpoints', metavar='DIR',
                    help='checkpoints path to save')

parser.add_argument('--train', default=False, type=bool,
                    help='train or val')
parser.add_argument('--in_memory', default=False, type=bool,
                    help='pre-loead the image to memory')

parser.add_argument('--epochs', default=210, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--eval_epoch', default=1, type=int,
                    help='every eval_epoch will evaluate')

parser.add_argument('--batch_size', default=32, type=int,
                    help='training batch size')
parser.add_argument('--resize', default=512, type=int,
                    help='resize the image')
parser.add_argument('--size', default=448, type=int,
                    help='the cropped image size')

parser.add_argument('--model_name', default='resnet50', type=str,
                    help='model name: resnet50 or resnet152')
parser.add_argument('--method',  default='BR', type=str,
                    help='method name: CE, ME, PC, RW, LA, BR')
parser.add_argument('--tau', default=0.5, type=float,
                    help='the weight for the other approach')
parser.add_argument('--pow', default=0.5, type=float,
                    help='the power of the weight')
parser.add_argument('--adj_type', default=1, type=int,
                    help='adjustment type')

parser.add_argument('--lr', '--learning-rate', default=0.0004, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=5.0e-5, type=float,
                    metavar='W', help='weight decay (default: 1.0e-5)')
parser.add_argument('--gpu_num', default=4, type=int,
                    help='GPU nums to use.')

parser.add_argument('--device',  default='cuda', type=str)


if __name__ == '__main__':
    from utils.main import main
    import numpy as np

    def run(args, times=1):
        accs = []
        for i in range(times):
            go = main(args=args, seed=i)
            acc = go.build()
            accs.append(acc.item())

        print('Average performance of {} times, Results: {:.2f}%, Std: {:.2f}%\n'
              .format(times, 100*np.mean(accs), 100*np.std(accs)))

    args = parser.parse_args()
    run(args, times=1)
