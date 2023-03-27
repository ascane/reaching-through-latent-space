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

from geco import GECO
from robot_state_dataset import RobotStateDataset
from training_utils import save_args, optimiser_to
from vae import VAE
from yaml_loader import YamlLoader

from sim.panda import Panda

FMAX = np.finfo('float').max

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

'''
Example command:
src$ python train_vae.py --c ../config/vae_config/panda_10k.yaml
'''

def main(args):

    # set up model directories
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    snapshot_dir = os.path.join(args.model_dir, 'snapshots')
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    # save run command
    save_args(args, run_dir=args.model_dir)
 
    # set up torch cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # set numpy random seed
    np.random.seed(args.eval_seed)

    # load dataset
    path = args.path_to_dataset
    train_loader = torch.utils.data.DataLoader(
        RobotStateDataset(path, train=0, train_data_name=args.train_data_name, test_data_name=args.test_data_name), 
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    val_dataset = RobotStateDataset(path, train=1, train_data_name=args.train_data_name, test_data_name=args.test_data_name)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    mean_train = val_dataset.get_mean_train()
    std_train = val_dataset.get_std_train()
    mean_train_tensor = torch.Tensor(mean_train).to(device)
    std_train_tensor = torch.Tensor(std_train).to(device)

    # set up model and optimiser
    model = VAE(args.input_dim, args.latent_dim, args.units_per_layer, args.num_hidden_layers)
    optimiser_vae = optim.Adam(model.parameters(), lr=args.lr_vae)

    geco = GECO(args.g_goal, args.g_lr, args.g_alpha, args.g_init, args.g_min, args.g_max, args.g_s)
    last_epoch_vae = 0

    # load and resume the latest checkpoint
    checkpoint_file_path = os.path.join(args.model_dir, "checkpoint")
    if os.path.exists(checkpoint_file_path):
        with open(checkpoint_file_path, 'r') as fp:
            checkpoint_path = fp.readline()
            # remove \n at the end
            if checkpoint_path.endswith('\n'):
                checkpoint_path = checkpoint_path[:-1]
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimiser_vae.load_state_dict(checkpoint['optimiser_vae_state_dict'])
            geco.load_state_dict(checkpoint['geco_state_dict'])
            last_epoch_vae = checkpoint['epoch_vae']
    
    model.to(device)
    optimiser_to(optimiser_vae, device)
    geco.to(device)

    # set up robo for computing FK
    robo = None
    if args.robo_name == 'panda':
        robo = Panda()
    elif args.robo_name == 'snake':
        robo = Snake()

    dof = robo.get_dof()
    robo.to(device)


    # set up tensorboard summary writer
    writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'runs'))

    def loss_function(xPrime, x, mu, logVar, train=True):

        mse_output = F.mse_loss(xPrime, x, reduction='sum') / args.batch_size

        # take the channelwise sum and then take the mean over batches
        KL_div = -0.5 * torch.mean(torch.sum(1 + logVar - mu.pow(2) - logVar.exp(), 1))

        if train:
            loss = geco.loss(mse_output, KL_div)
        else:
            loss = geco.geco_lambda * mse_output + KL_div

        elbo = mse_output + KL_div
        
        return loss, KL_div, mse_output, elbo

    def train(epoch):

        model.train()
        train_loss = 0
        train_kl = 0
        train_ms = 0
        train_elbo = 0
        train_geco_lambda = 0

        for batch_index, (data, _) in enumerate(train_loader):

            data = data.to(device)

            optimiser_vae.zero_grad()

            xPrime_batch, mu_batch, logVar_batch = model(data)

            train_geco_lambda += geco.geco_lambda.detach().item() * len(data) / len(train_loader.dataset)

            loss = loss_function(xPrime_batch, data, mu_batch, logVar_batch)
            loss[0].backward()

            train_loss += loss[0].detach().item() * len(data) / len(train_loader.dataset)
            train_kl += loss[1].detach().item() * len(data) / len(train_loader.dataset)
            train_ms += loss[2].detach().item() * len(data) / len(train_loader.dataset)
            train_elbo += loss[3].detach().item() * len(data) / len(train_loader.dataset)
            
            optimiser_vae.step()

            if batch_index % args.log_interval == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlambda: {:.010f}'.format(
                    epoch, batch_index * args.batch_size + len(data), len(train_loader.dataset),
                    100. * (batch_index + 1) / len(train_loader),
                    loss[0].item(),
                    geco.geco_lambda.item()))

        logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

        return train_loss, train_kl, train_ms, train_elbo, train_geco_lambda


    def val(epoch):

        model.eval()
        val_loss = 0
        val_kl = 0
        val_ms = 0
        val_elbo = 0
        val_geco_lambda = 0

        with torch.no_grad():
            for data, _ in val_loader:

                data = data.to(device)

                optimiser_vae.zero_grad()

                xPrime_batch, mu_batch, logVar_batch = model(data)

                val_geco_lambda += geco.geco_lambda.detach().item() * len(data) / len(val_loader.dataset)

                loss = loss_function(xPrime_batch, data, mu_batch, logVar_batch, train=False)

                val_loss += loss[0].detach().item() * len(data) / len(val_loader.dataset)
                val_kl += loss[1].detach().item() * len(data) / len(val_loader.dataset)
                val_ms += loss[2].detach().item() * len(data) / len(val_loader.dataset)
                val_elbo += loss[3].detach().item() * len(data) / len(val_loader.dataset)
                
            logging.info('====> Val set loss: {:.4f}'.format(val_loss))

            return val_loss, val_kl, val_ms, val_elbo, val_geco_lambda

    def xPrime_batch_to_fk_loss(xPrime_batch):

        # denormalise xPrime_batch
        batch_size = xPrime_batch.shape[0]
        xPrime_batch = xPrime_batch * std_train_tensor[:, :model.input_dim].repeat(batch_size, 1)
        xPrime_batch = xPrime_batch + mean_train_tensor[:, :model.input_dim].repeat(batch_size, 1)

        xPrime_fk_batch = robo.FK(xPrime_batch[:, :-3], device, rad=True)
        FK_loss = torch.mean(torch.cdist(xPrime_fk_batch.unsqueeze(1), xPrime_batch[:, -3:].unsqueeze(1)).squeeze())

        # FK_loss = torch.tensor(0.0).to(device)
        # for i in range(batch_size):
        #     FK_loss += F.mse_loss(robo.compute_fk(xPrime_batch[i, :-3]).squeeze().to(device), xPrime_batch[i, -3:])

        return FK_loss

    def sample_consistency_posterior():
        
        model.eval()
        fk_loss = 0

        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            xPrime_batch = model.get_reconstruction(x_batch)
            loss = xPrime_batch_to_fk_loss(xPrime_batch)
            fk_loss += loss.detach().item() * len(x_batch) / len(val_loader.dataset)

        logging.info('====> Sample consistency posterior: {:.4f}'.format(fk_loss))
    
        return fk_loss

    def sample_consistency_prior():
        
        model.eval()

        # sample from prior
        np_z_batch = np.random.multivariate_normal(mean=[0]*args.latent_dim, cov=np.identity(args.latent_dim), size=args.samples)
        z_batch = torch.tensor(np_z_batch, dtype=torch.float32)
        z_batch = z_batch.to(device)

        xPrime_batch = model.get_recon_from_latent(z_batch)

        fk_loss = xPrime_batch_to_fk_loss(xPrime_batch).detach().item()

        logging.info('====> Sample consistency prior: {:.4f}'.format(fk_loss))

        return fk_loss


    def am_min_distances(use_fk_loss=False):

        logging.info('====> Computing AM min distances...')
        torch.autograd.set_detect_anomaly(True)

        model.eval()
        min_distances = np.array([FMAX] * args.am_samples)

        for i in range(args.am_samples):
            x, goal_xyz = val_dataset[i]
            x = x.to(device)
            goal_xyz = goal_xyz.to(device)

            # denormalise goal_xyz
            goal_xyz = goal_xyz * std_train_tensor[:, model.input_dim:model.input_dim + 3]
            goal_xyz = goal_xyz + mean_train_tensor[:, model.input_dim:model.input_dim + 3]

            # pass input through model to get latent features
            z = model.get_features(x).detach().requires_grad_(True)

            optimiser_am = optim.Adam([z], lr=args.am_lr)

            # gradient descent in latent space
            for _ in range(1, args.am_steps + 1):

                # zero grad optimiser
                optimiser_am.zero_grad()

                # compute model output
                xPrime = model.decoder(z)

                # denormalise output
                xPrime = xPrime * std_train_tensor[:, :model.input_dim]
                xPrime = xPrime + mean_train_tensor[:, :model.input_dim]
                jposPrime = xPrime[:, :dof]
                ee_xyz_prime = xPrime[:, dof:]
                ee_xyz_prime_fk = robo.FK(jposPrime, device, rad=True, joint_limit=True)
                # ee_xyz_prime_fk = robo.compute_fk(jposPrime[0]).to(device)

                dist_ee_goal = torch.dist(ee_xyz_prime, goal_xyz)
                dist_ee_goal_fk = torch.dist(ee_xyz_prime_fk, goal_xyz)
                
                min_distances[i] = min(dist_ee_goal_fk.detach().item(), min_distances[i])

                if use_fk_loss:
                    total_loss = dist_ee_goal_fk
                else:
                    total_loss = dist_ee_goal
                total_loss.backward()
                optimiser_am.step()

        if use_fk_loss:
            logging.info('====> Using FK loss')
        else:
            logging.info('====> Not using FK loss')
        logging.info('====> AM min distances: %s' % min_distances[:10].tolist())
        logging.info('====> AM mean min distances: %f' % np.mean(min_distances))

        return min_distances

    def auc(values, parts=100, value_max=0.1):
        '''Area under curve'''
        area = 0
        size = len(values)
        values = sorted(values)
        success_count = 0
        for i in range(parts + 1):
            threshold = value_max * i / parts
            while success_count < size and values[success_count] <= threshold:
                success_count += 1
            area += float(success_count) / size  # height of each discretised bar
        area *= value_max / parts  # width of each discretised bar

        area_p = area / value_max  # total_area = 1 (height) * value_max (width)

        logging.info('====> AUC: %f' % area_p)

        return area_p


    for epoch in range(last_epoch_vae + 1, args.epochs_vae + 1):

        train_loss, train_kl, train_ms, train_elbo, train_geco_lambda = train(epoch)
        val_loss, val_kl, val_ms, val_elbo, val_geco_lambda = val(epoch)

        writer.add_scalar('val_loss', val_loss, global_step=epoch)
        writer.add_scalar('val_kl', val_kl, global_step=epoch)
        writer.add_scalar('val_mse', val_ms, global_step=epoch)
        writer.add_scalar('val_elbo', val_elbo, global_step=epoch)
        writer.add_scalar('val_geco_lambda', val_geco_lambda, global_step=epoch)

        writer.add_scalar('training_loss', train_loss, global_step=epoch)
        writer.add_scalar('training_kl', train_kl, global_step=epoch)
        writer.add_scalar('training_mse', train_ms, global_step=epoch)
        writer.add_scalar('training_elbo', train_elbo, global_step=epoch)
        writer.add_scalar('training_geco_lambda', train_geco_lambda, global_step=epoch)

        if epoch % args.save_every == 0:
            # save checkpoint to snapshot dir
            checkpoint_name = 'model.ckpt-%06d.pt' % (epoch)
            checkpoint_path = os.path.join(snapshot_dir, checkpoint_name)
            torch.save({
                'epoch_vae': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_vae_state_dict': optimiser_vae.state_dict(),
                'geco_state_dict': geco.state_dict(),
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
            if args.samples > 0:
                # -- compute sample consistency from posterior and from prior
                l2_dist_posterior = sample_consistency_posterior()
                l2_dist_prior = sample_consistency_prior()
                am_min_dist = am_min_distances(use_fk_loss=False)
                # am_min_dist_fk = am_min_distances(use_fk_loss=True)
                am_auc = auc(am_min_dist, args.am_auc_parts, args.am_auc_max)
                # am_auc_fk = auc(am_min_dist_fk, args.am_auc_parts, args.am_auc_max)

                writer.add_scalar('l2_dist_posterior', l2_dist_posterior, global_step=epoch)
                writer.add_scalar('l2_dist_prior', l2_dist_prior, global_step=epoch)
                writer.add_scalar('am_mean_min_dist', np.mean(am_min_dist), global_step=epoch)
                # writer.add_scalar('am_mean_min_dist_fk', np.mean(am_min_dist_fk), global_step=epoch)
                writer.add_scalar('am_auc', am_auc, global_step=epoch)
                # writer.add_scalar('am_auc_fk', am_auc_fk, global_step=epoch)

                snapshot_index[checkpoint_name] = {
                    'path': checkpoint_path,
                    'train_loss': train_loss,
                    'train_kl': train_kl,
                    'train_mse': train_ms,
                    'train_elbo': train_elbo,
                    'train_geco_lambda': train_geco_lambda,
                    'val_loss': val_loss,
                    'val_kl': val_kl,
                    'val_mse': val_ms,
                    'val_elbo': val_elbo,
                    'val_geco_lambda': val_geco_lambda,
                    'epoch_vae': epoch,
                    'l2_dist_posterior' : l2_dist_posterior,
                    'l2_dist_prior': l2_dist_prior,
                    'am_mean_min_dist': np.mean(am_min_dist),
                    # 'am_mean_min_dist_fk': np.mean(am_min_dist_fk),
                    'am_auc': am_auc,
                    # 'am_auc_fk': am_auc_fk,
                    'am_min_dist': am_min_dist.tolist(),
                    # 'am_min_dist_fk': am_min_dist_fk.tolist(),
                }
            else:
                snapshot_index[checkpoint_name] = {
                    'path': checkpoint_path,
                    'train_loss': train_loss,
                    'train_kl': train_kl,
                    'train_mse': train_ms,
                    'train_elbo': train_elbo,
                    'train_geco_lambda': train_geco_lambda,
                    'val_loss': val_loss,
                    'val_kl': val_kl,
                    'val_mse': val_ms,
                    'val_elbo': val_elbo,
                    'val_geco_lambda': val_geco_lambda,
                    'epoch_vae': epoch,
                }
            snapshot_info[checkpoint_name] = snapshot_index[checkpoint_name]
            # --- remove worst snapshot, if save slots are exceeded
            if args.num_best_ckpt > 0 and len(snapshot_index) > args.num_best_ckpt:
                ckpt_by_am_auc = list(snapshot_index.items())
                ckpt_by_am_auc.sort(key=lambda t: t[1]['am_auc'])
                worst_ckpt, _ = ckpt_by_am_auc[0]  # get worst checkpoint name
                worst_ckpt_path = snapshot_index[worst_ckpt]['path']
                os.remove(worst_ckpt_path)
                worst_info = snapshot_index.pop(worst_ckpt)
                logging.info(">>> Removed worst snapshot (step=%d; am_auc=%.06f): %s" % \
                    (worst_info['epoch_vae'], worst_info['am_auc'], worst_info['path']))  # DEBUG
            # --- save snapshot index
            with open(snapshot_index_file, 'w') as fp:
                json.dump(snapshot_index, fp, indent=2, sort_keys=True)
            logging.info(">>> Saved snapshot index: %s" % snapshot_index_file)
            # --- save snapshot info
            with open(snapshot_info_file, 'w') as fp:
                json.dump(snapshot_info, fp, indent=2, sort_keys=True)
            logging.info(">>> Saved snapshot info: %s" % snapshot_info_file)
            # --- save current last checkpoint
            current_last_checkpoint_path = os.path.join(args.model_dir, checkpoint_name)
            torch.save({
                'epoch_vae': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_vae_state_dict': optimiser_vae.state_dict(),
                'geco_state_dict': geco.state_dict(),
            }, current_last_checkpoint_path)
            logging.info(">>> Saved current last checkpoint: %s" % current_last_checkpoint_path)
            # --- remove previous last checkpoint
            if os.path.isfile(checkpoint_file_path):
                with open(checkpoint_file_path, 'r') as fp:
                    previous_last_checkpoint_path = fp.read()
                    if previous_last_checkpoint_path.endswith('\n'):
                        previous_last_checkpoint_path = previous_last_checkpoint_path[:-1]
                    os.remove(previous_last_checkpoint_path)
            # --- save checkpoint path to checkpoint file
            with open(checkpoint_file_path, 'w') as fp:
                fp.write(current_last_checkpoint_path)

    # save last checkpoint to model dir
    last_epoch_vae = max(last_epoch_vae, args.epochs_vae)
    # --- save checkpoint path to checkpoint file
    checkpoint_name = 'model.ckpt-%06d.pt' % (last_epoch_vae)
    checkpoint_path = os.path.join(args.model_dir, checkpoint_name)
    with open(checkpoint_file_path, 'w') as fp:
        fp.write(checkpoint_path)
    # ---save checkpoint to checkpoint path
    torch.save({
        'epoch_vae': last_epoch_vae,
        'model_state_dict': model.state_dict(),
        'optimiser_vae_state_dict': optimiser_vae.state_dict(),
        'geco_state_dict': geco.state_dict(),
    }, checkpoint_path)
    logging.info(">>> Saved checkpoint: %s" % checkpoint_path)

    writer.close()


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser(description='VAE 3D')
    args = YamlLoader(parser).return_args()

    # for backward compatibility
    if not hasattr(args, 'robo_name'):
        args.__dict__['robo_name'] = 'panda'

    main(args)