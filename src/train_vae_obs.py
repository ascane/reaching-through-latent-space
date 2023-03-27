from __future__ import print_function
import argparse
import json
import logging
import numpy as np
import os
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from robot_obs_dataset import RobotObstacleDataset
from training_utils import save_args, optimiser_to
from vae_obs import VAEObstacleBCE
from yaml_loader import YamlLoader

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

'''
Example command:
src$ python train_vae_obs.py --c ../config/vae_obs_config/panda_10k.yaml
'''

def main(args):

    # set up model directories
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    snapshot_dir = os.path.join(args.model_dir, 'snapshots_obs')
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    # load model training args
    with open(args.vae_run_cmd_path) as json_file:
        data = json.load(json_file)
        training_args = data['parsed_args']

    args.__dict__['input_dim'] = training_args['input_dim']
    args.__dict__['latent_dim'] = training_args['latent_dim']
    args.__dict__['units_per_layer'] = training_args['units_per_layer']
    args.__dict__['num_hidden_layers'] = training_args['num_hidden_layers']
    args.__dict__['free_space_train_name'] = training_args['train_data_name']
    args.__dict__['free_space_test_name'] = training_args['test_data_name']

    # save run command
    save_args(args, run_dir=args.model_dir)
 
    # set up torch cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # load dataset
    path = args.path_to_dataset
    train_loader = torch.utils.data.DataLoader(
        RobotObstacleDataset(path, train=0, robo_name=args.robo_name, train_data_name=args.train_data_name, test_data_name=args.test_data_name, \
                             free_space_train_name=args.free_space_train_name, free_space_test_name=args.free_space_test_name), 
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    val_dataset = RobotObstacleDataset(path, train=1, robo_name=args.robo_name, train_data_name=args.train_data_name, test_data_name=args.test_data_name, \
                                       free_space_train_name=args.free_space_train_name, free_space_test_name=args.free_space_test_name)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # set up model and optimiser
    model = VAEObstacleBCE(args.input_dim, args.latent_dim, args.units_per_layer, args.num_hidden_layers)
    
    # load pretrained VAE model dict
    checkpoint_vae = torch.load(args.pretrained_checkpoint_path)
    pretrained_dict = checkpoint_vae['model_state_dict']
    ## get model dict
    model_dict = model.state_dict()
    ## filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    ## overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    ## load the new state dict
    model.load_state_dict(model_dict)

    obs_params = list(model.fc32.parameters()) + list(model.fc42.parameters())
    for fc in model.fc_obs:
        obs_params += list(fc.parameters())
    optimiser_obs = optim.Adam(obs_params, lr=args.lr_obs)

    last_epoch_obs = 0
    last_epoch_vae = checkpoint_vae['epoch_vae']

    # load and resume the latest checkpoint
    checkpoint_file_path = os.path.join(args.model_dir, "checkpoint_obs")
    if os.path.exists(checkpoint_file_path):
        with open(checkpoint_file_path, 'r') as fp:
            checkpoint_path = fp.readline()
            # remove \n at the end
            if checkpoint_path.endswith('\n'):
                checkpoint_path = checkpoint_path[:-1]
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimiser_obs.load_state_dict(checkpoint['optimiser_obs_state_dict'])
            last_epoch_obs = checkpoint['epoch_obs']
    
    model.to(device)
    optimiser_to(optimiser_obs, device)

    # set up tensorboard summary writer
    writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'runs_obs'))


    def eval_collision_prediction_rate(obs_logit, obs_label):
        obs_prob_np = torch.sigmoid(obs_logit).cpu().detach().numpy()
        obs_label_np = obs_label.cpu().detach().numpy()
        count = (obs_label_np[obs_prob_np >= 0.5] == 1).sum()
        count += (obs_label_np[obs_prob_np < 0.5] == 0).sum()
        return float(count) / obs_prob_np.shape[0]

    def loss_function_obs(obs_logit, obs_label):
        bce_output = F.binary_cross_entropy_with_logits(obs_logit, obs_label, reduction='mean') + 1.0e-6
        return bce_output
    
    def train(epoch):
        model.train()
        train_loss = 0
        train_collision_pred_rate = 0

        for batch_index, (data, obs_state, obs_label) in enumerate(train_loader):

            data = data.to(device)
            obs_state = obs_state.to(device)
            obs_label = obs_label.to(device)

            optimiser_obs.zero_grad()

            _, _, _, obs_logit = model(data, obs_state)

            collision_pred_rate = eval_collision_prediction_rate(obs_logit, obs_label)

            loss = loss_function_obs(obs_logit, obs_label)
            loss.backward()

            train_loss += loss.item() * len(data) / len(train_loader.dataset)
            train_collision_pred_rate +=  collision_pred_rate * len(data) / len(train_loader.dataset)
            
            optimiser_obs.step()

            if batch_index % args.log_interval == 0:
                logging.info('Obs Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPred rate: {:.2f}{}'.format(
                    epoch, batch_index * len(data), len(train_loader.dataset),
                    100. * batch_index / len(train_loader),
                    loss.item(),
                    collision_pred_rate * 100,
                    chr(37)))

        logging.info('====> Obs Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss))

        return train_loss, train_collision_pred_rate

    def val(epoch):
        model.eval()
        val_loss = 0
        val_collision_pred_rate = 0

        with torch.no_grad():
            for batch_index, (data, obs_state, obs_label) in enumerate(val_loader):

                data = data.to(device)
                obs_state = obs_state.to(device)
                obs_label = obs_label.to(device)

                optimiser_obs.zero_grad()

                _, _, _, obs_logit = model(data, obs_state)

                collision_pred_rate = eval_collision_prediction_rate(obs_logit, obs_label)

                loss = loss_function_obs(obs_logit, obs_label)

                val_loss += loss.item() * len(data) / len(val_loader.dataset)
                val_collision_pred_rate +=  collision_pred_rate * len(data) / len(val_loader.dataset)

            logging.info('====> Obs Val set loss: {:.4f}\tPred rate: {:.2f}{}'.format(val_loss, val_collision_pred_rate * 100, chr(37)))

            return val_loss, val_collision_pred_rate


    for epoch in range(last_epoch_obs + 1, args.epochs_obs + 1):
        train_loss_obs, train_collision_pred_rate = train(epoch)
        val_loss_obs, val_collision_pred_rate = val(epoch)

        writer.add_scalar('val_loss_obs', val_loss_obs, global_step=epoch)
        writer.add_scalar('val_collision_pred_rate', val_collision_pred_rate, global_step=epoch)
        writer.add_scalar('training_loss_obs', train_loss_obs, global_step=epoch)
        writer.add_scalar('train_collision_pred_rate', train_collision_pred_rate, global_step=epoch)

        if epoch % args.save_every == 0:
            # save checkpoint to snapshot dir
            checkpoint_name = 'model.ckpt-%06d-%06d.pt' % (last_epoch_vae, epoch)
            checkpoint_path = os.path.join(snapshot_dir, checkpoint_name)
            torch.save({
                'epoch_vae': last_epoch_vae,
                'epoch_obs': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_obs_state_dict': optimiser_obs.state_dict(),
            }, checkpoint_path)
            logging.info(">>> Saved checkpoint: %s" % checkpoint_path)

            # save snapshot indices and their respective losses
            snapshot_index_file = os.path.join(snapshot_dir, 'snapshot_index.json')
            if os.path.exists(snapshot_index_file):  # load snapshot index
                with open(snapshot_index_file, 'r') as fp:
                    snapshot_index = json.load(fp)
            else:
                snapshot_index = {}
            # save snapshot info (keep same info as snapshot_index)
            snapshot_info_file = os.path.join(snapshot_dir, 'snapshot_info.json')
            if os.path.exists(snapshot_info_file):  # load snapshot index
                with open(snapshot_info_file, 'r') as fp:
                    snapshot_info = json.load(fp)
            else:
                snapshot_info = {}
            # --- add current snapshot info
            snapshot_index[checkpoint_name] = {
                'path': checkpoint_path,
                'train_loss_obs': train_loss_obs,
                'val_loss_obs': val_loss_obs,
                'train_collision_pred_rate': train_collision_pred_rate,
                'val_collision_pred_rate' : val_collision_pred_rate,
                'epoch_vae': last_epoch_vae,
                'epoch_obs': epoch,
            }
            snapshot_info[checkpoint_name] = snapshot_index[checkpoint_name]
            # --- remove worst snapshot, if save slots are exceeded
            if args.num_best_ckpt > 0 and len(snapshot_index) > args.num_best_ckpt:
                ckpt_by_val_collision_pred_rate = list(snapshot_index.items())
                ckpt_by_val_collision_pred_rate.sort(key=lambda t: t[1]['val_collision_pred_rate'])
                worst_ckpt, _ = ckpt_by_val_collision_pred_rate[0]  # get worst checkpoint name
                worst_ckpt_path = snapshot_index[worst_ckpt]['path']
                os.remove(worst_ckpt_path)
                worst_info = snapshot_index.pop(worst_ckpt)
                logging.info(">>> Removed worst snapshot (step=%d; val_collision_pred_rate=%.06f): %s" % \
                    (worst_info['epoch_obs'], worst_info['val_collision_pred_rate'], worst_info['path']))  # DEBUG
            # --- save snapshot index
            with open(snapshot_index_file, 'w') as fp:
                json.dump(snapshot_index, fp, indent=2, sort_keys=True)
            logging.info(">>> Saved snapshot index: %s" % snapshot_index_file)
            # --- save snapshot info
            with open(snapshot_info_file, 'w') as fp:
                json.dump(snapshot_info, fp, indent=2, sort_keys=True)
            logging.info(">>> Saved snapshot info: %s" % snapshot_info_file)
            # --- save checkpoint path to checkpoint file
            with open(checkpoint_file_path, 'w') as fp:
                fp.write(checkpoint_path)
        
    # save last checkpoint to model dir
    last_epoch_obs = max(last_epoch_obs, args.epochs_obs)
    # --- save checkpoint path to checkpoint file
    checkpoint_name = 'model.ckpt-%06d-%06d.pt' % (last_epoch_vae, last_epoch_obs)
    checkpoint_path = os.path.join(args.model_dir, checkpoint_name)
    with open(checkpoint_file_path, 'w') as fp:
        fp.write(checkpoint_path)
    # ---save checkpoint to checkpoint path
    torch.save({
        'epoch_vae': last_epoch_vae,
        'epoch_obs': last_epoch_obs,
        'model_state_dict': model.state_dict(),
        'optimiser_obs_state_dict': optimiser_obs.state_dict(),
    }, checkpoint_path)
    logging.info(">>> Saved checkpoint: %s" % checkpoint_path)

    writer.close()



if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser(description='VAE Obstacle 3D')
    args = YamlLoader(parser).return_args()

    # for backward compatibility
    if not hasattr(args, 'robo_name'):
        args.__dict__['robo_name'] = 'panda'

    main(args)
