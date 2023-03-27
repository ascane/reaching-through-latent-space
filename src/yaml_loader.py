import argparse
import yaml


class YamlLoader:

    def __init__(self, parse):

        self.parser = parse

    def return_args(self):

        self.parser.add_argument('--config-file', dest='config_file', type=argparse.FileType(mode='r'),
                                 default='./config_files/default.yaml', metavar='c',
                                 help='path to Config yaml file')
        self.parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                                 help='input batch size for training (default: 128)')
        self.parser.add_argument('--epochs', type=int, default=6, metavar='N',
                                 help='number of epochs to train (default: 10)')
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')
        self.parser.add_argument('--log-grads', action='store_true', default=False,
                                 help='log gradients to tensorboard but accept speed reduction')
        self.parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
        self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                 help='how many batches to wait before logging training status')

        args = self.parser.parse_args()

        if args.config_file != '':

            yaml_data = yaml.safe_load(args.config_file)
            delattr(args, 'config_file')
            arg_dict = args.__dict__
            for key, value in yaml_data.items():

                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value

        return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VAE MNIST Example')

    yaml_loader = YamlLoader(parser)

    print(yaml_loader.return_args())
