import os
import copy
import logging
import time
import torch

import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from sklearn.metrics import roc_auc_score
from Models.NextStepModels import Pix2PixModel


def train(model, datamodule, args):
    train_dataloader, val_dataloader, test_dataloader = datamodule.train_dataloader(), datamodule.val_dataloader(), \
                                                        datamodule.test_dataloader()
    checkpoint = {'model': None, 'val_loss': 1e10, 'val_acc': None, 'val_auc': None, 'test_loss': None,
                  'test_acc': None, 'test_auc': None, 'iter': None}
    epoch, iteration = 0, 0
    model.train()
    if not args.debug:
        for epoch in range(1, args.num_epochs + 1):
            epoch_loss = 0.
            for itr, batch in enumerate(train_dataloader, 0):
                model.train()
                x, y = batch
                x, y = x.to(args.device), y.to(args.device)
                y_pred = model(x)
                loss = args.criterion(y_pred, y)
                args.optimizer.zero_grad()
                loss.backward()
                args.optimizer.step()
                args.writer.add_scalar('Train/Loss_Iter', loss, iteration)
                epoch_loss += loss
                iteration += 1
                if (itr + 1) % args.num_test_iters == 0:
                    eval_val = test(model, val_dataloader, args)
                    eval_test = test(model, test_dataloader, args)
                    args.writer.add_scalar('Val/Loss_Iter', eval_val['loss'], iteration)
                    args.writer.add_scalar('Val/Acc_Iter', eval_val['acc'], iteration)
                    args.writer.add_scalar('Val/AUC_Iter', eval_val['auc'], iteration)
                    args.writer.add_scalar('Test/Loss_Iter', eval_test['loss'], iteration)
                    args.writer.add_scalar('Test/Acc_Iter', eval_test['acc'], iteration)
                    args.writer.add_scalar('Test/AUC_Iter', eval_test['auc'], iteration)
                    logging.warning(
                        'Epoch: {}/{} \t Iter: {}/{} \t Loss: {:.4f} \t Val_Acc: {:.4f} \t Val_AUC: {:.4f} \t'
                        ' Test_Acc: {:.4f} \t Test_AUC: {:.4f}'.
                            format(epoch, args.num_epochs, itr + 1, len(train_dataloader), loss.item(), eval_val['acc'],
                                   eval_val['auc'], eval_test['acc'], eval_test['auc']))
                    if eval_val['loss'] < checkpoint['val_loss']:
                        checkpoint['model'] = copy.deepcopy(model.state_dict())
                        checkpoint['val_acc'], checkpoint['val_loss'], checkpoint['val_auc'] = eval_val['acc'], \
                                                                                               eval_val['loss'], \
                                                                                               eval_val['auc']
                        checkpoint['test_acc'], checkpoint['test_loss'], checkpoint['test_auc'] = eval_test['acc'], \
                                                                                                  eval_test['loss'], \
                                                                                                  eval_test['auc']
                        checkpoint['iter'] = iteration
                else:
                    logging.warning('Epoch: {}/{} \t Iter: {}/{} \t Loss: {:.4f}'.
                                    format(epoch, args.num_epochs, itr + 1, len(train_dataloader), loss.item()))

            # Testing at the end of epoch.
            epoch_loss = epoch_loss / len(train_dataloader)
            logging.warning('Evaluating Network at the end of epoch #{}:'.format(epoch))
            eval_val = test(model, val_dataloader, args)
            eval_test = test(model, test_dataloader, args)
            args.writer.add_scalar('Train/Loss_Epoch', epoch_loss, epoch)
            args.writer.add_scalar('Val/Loss_Epoch', eval_val['loss'], epoch)
            args.writer.add_scalar('Val/Acc_Epoch', eval_val['acc'], epoch)
            args.writer.add_scalar('Val/AUC_Epoch', eval_val['auc'], epoch)
            args.writer.add_scalar('Test/Loss_Epoch', eval_test['loss'], epoch)
            args.writer.add_scalar('Test/Acc_Epoch', eval_test['acc'], epoch)
            args.writer.add_scalar('Test/AUC_Epoch', eval_test['auc'], epoch)
            logging.warning('Epoch: {}/{} \t Epoch_Loss: {:.4f} \t Val_Acc: {:.4f} \t Val_AUC: {:.4f} \t '
                            'Test_Acc: {:.4f} \t Test_AUC: {:.4f}'.
                            format(epoch, args.num_epochs, epoch_loss.item(), eval_val['acc'], eval_val['auc'],
                                   eval_test['acc'], eval_test['auc']))
            if eval_val['loss'] < checkpoint['val_loss']:
                checkpoint['model'] = copy.deepcopy(model.state_dict())
                checkpoint['val_acc'], checkpoint['val_loss'], checkpoint['val_auc'] = eval_val['acc'], eval_val[
                    'loss'], eval_val['auc']
                checkpoint['test_acc'], checkpoint['test_loss'], checkpoint['test_auc'] = eval_test['acc'], eval_test[
                    'loss'], eval_test['auc']
                checkpoint['iter'] = iteration

        return checkpoint

    else:
        model.train()
        train_batch = next(iter(train_dataloader))
        x, y = train_batch[0].to(args.device), train_batch[1].to(args.device)
        for epoch in range(args.num_epochs):
            model.train()
            y_pred = model(x)
            loss = args.criterion(y_pred, y)
            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()
            epoch_loss = loss
            eval_val = test(model, val_dataloader, args)
            eval_test = test(model, test_dataloader, args)
            args.writer.add_scalar('Train/Loss_Epoch', epoch_loss.item(), epoch)
            args.writer.add_scalar('Val/Loss_Epoch', eval_val['loss'], epoch)
            args.writer.add_scalar('Val/Acc_Epoch', eval_val['acc'], epoch)
            args.writer.add_scalar('Val/AUC_Epoch', eval_val['auc'], epoch)
            args.writer.add_scalar('Test/Loss_Epoch', eval_test['loss'], epoch)
            args.writer.add_scalar('Test/Acc_Epoch', eval_test['acc'], epoch)
            args.writer.add_scalar('Test/AUC_Epcoh', eval_test['auc'], epoch)
            logging.warning('Epoch: {}/{} \t Epoch_Loss: {:.4f} \t Val_Acc: {:.4f} \t Val_AUC: {:.4f} \t '
                            'Test_Acc: {:.4f} \t Test_AUC: {:.4f}'.
                            format(epoch, args.num_epochs, epoch_loss.item(), eval_val['acc'], eval_val['auc'],
                                   eval_test['acc'], eval_test['auc']))
            if eval_val['loss'] < checkpoint['val_loss']:
                checkpoint['model'] = copy.deepcopy(model.state_dict())
                checkpoint['val_acc'], checkpoint['val_loss'], checkpoint['val_auc'] = eval_val['acc'], eval_val[
                    'loss'], eval_val['auc']
                checkpoint['test_acc'], checkpoint['test_loss'], checkpoint['test_auc'] = eval_test['acc'], eval_test[
                    'loss'], eval_test['auc']
        return checkpoint


def test(model, test_dataloader, args):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_preds = []
    y_true = []
    with torch.no_grad():
        for data in test_dataloader:
            x, y = data
            x, y = x.to(args.device), y.to(args.device)
            y_pred = model(x)
            y_preds.append(y_pred)
            y_true.append(y)
            loss = args.criterion(y_pred, y)
            total_loss += loss
            _, label_pred = torch.max(y_pred, dim=1)
            total += y.shape[0]
            correct += (label_pred == y).sum().item()
    total_loss = total_loss / total
    accuracy = correct / total
    y_preds = torch.softmax(torch.stack(y_preds, dim=0).squeeze(), dim=1)
    y_true = torch.stack(y_true).squeeze()
    if not args.debug:
        if args.num_class == 2:
            auc = roc_auc_score(y_true.cpu().numpy(), (y_preds[:, 1].squeeze()).cpu().numpy())
        else:
            auc = roc_auc_score(y_true.cpu().numpy(), y_preds.cpu().numpy(), multi_class='ovo')
    else:
        auc = 0.
    model.train()
    return {'loss': total_loss, 'acc': accuracy, 'auc': auc}


def train_pix2pix(model, datamodule, args):
    train_dataloader, val_dataloader, test_dataloader = \
        datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()
    model.train()
    iteration = 0
    train_b, val_b, test_b = prepare_random_batch(train_dataloader, val_dataloader, test_dataloader, args)
    for epoch in range(args.epoch_count, args.n_epochs + args.n_epochs_decay + 1):
        epoch_start = time.time()
        model.update_learning_rate()
        for itr, batch in enumerate(train_dataloader, 0):
            # if itr == 1:
            #     break
            model.set_input(batch[0])
            model.optimize_parameters()
            iteration += 1
            losses = model.get_current_losses()
            if (itr + 1) % args.print_freq == 0:
                args.writer = add_losses_to_writer(args.writer, losses, iteration=iteration, on_epoch=False)
                logging.warning(prepare_logging_string(losses, epoch, args.n_epochs + args.n_epochs_decay, itr + 1,
                                                       len(train_dataloader)))
        logging.warning('End of Epoch #{} \t time taken: {:.4f} seconds.'.format(epoch, time.time() - epoch_start))
        if epoch % args.num_save_epochs == 0:
            logging.warning('Saving models at the end of epoch {}'.format(epoch))
            model.save_networks(epoch)
            model.to(model.device)
            args.epoch = epoch
            logging.warning('Evaluating models at the end of epoch {}'.format(epoch))
            evaluate_pix2pix(train_b, val_b, test_b, epoch, args)


def evaluate_pix2pix(train_b, val_b, test_b, epoch, args):
    n = args.num_save_image
    model = Pix2PixModel(args, False).to(args.device)
    model.setup(args)
    if args.test_mode_eval:
        model.eval()
    if epoch == args.num_save_epochs:
        plt.figure()
        plt.subplot(211)
        plt.axis("off")
        plt.title("Training Images First Step")
        plt.imshow(
            np.transpose(vutils.make_grid(train_b[0][0][:n].squeeze(), padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.subplot(212)
        plt.axis("off")
        plt.title("Training Images Second Step")
        plt.imshow(
            np.transpose(vutils.make_grid(train_b[0][1][:n].squeeze(), padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig(os.path.join(args.stdout, 'train_images.png'))
        plt.close()

        plt.figure()
        plt.subplot(211)
        plt.axis("off")
        plt.title("Validation Images First Step")
        plt.imshow(
            np.transpose(vutils.make_grid(val_b[0][0][:n].squeeze(), padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.subplot(212)
        plt.axis("off")
        plt.title("Validation Images Second Step")
        plt.imshow(
            np.transpose(vutils.make_grid(val_b[0][1][:n].squeeze(), padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig(os.path.join(args.stdout, 'val_images.png'))
        plt.close()

        plt.figure()
        plt.subplot(211)
        plt.axis("off")
        plt.title("Test Images First Step")
        plt.imshow(
            np.transpose(vutils.make_grid(test_b[0][0][:n].squeeze(), padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.subplot(212)
        plt.axis("off")
        plt.title("Test Images Second Step")
        plt.imshow(
            np.transpose(vutils.make_grid(test_b[0][1][:n].squeeze(), padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig(os.path.join(args.stdout, 'test_images.png'))
        plt.show()
        plt.close()

    x_tilde_train, x_tilde_val, x_tilde_test = [], [], []
    for i in range(n):
        model.set_input((torch.unsqueeze(train_b[0][0][i], 0), torch.unsqueeze(train_b[0][1][i], 0)))
        model.test()
        x_tilde_train.append(model.fake_B)

        model.set_input((torch.unsqueeze(val_b[0][0][i], 0), torch.unsqueeze(val_b[0][1][i], 0)))
        model.test()
        x_tilde_val.append(model.fake_B)

        model.set_input((torch.unsqueeze(test_b[0][0][i], 0), torch.unsqueeze(test_b[0][1][i], 0)))
        model.test()
        x_tilde_test.append(model.fake_B)
    x_tilde_train, x_tilde_val, x_tilde_test = torch.stack(x_tilde_train).squeeze(), torch.stack(x_tilde_val).squeeze(), \
                                               torch.stack(x_tilde_test).squeeze()

    plt.figure()
    plt.axis("off")
    plt.title("Training Generated Images Epoch {}".format(epoch))
    plt.imshow(
        np.transpose(vutils.make_grid(x_tilde_train.squeeze().cpu(), padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(args.stdout, 'train_gen_images_ep_{}.png'.format(epoch)))
    plt.close()

    plt.figure()
    plt.axis("off")
    plt.title("Validation Generated Images Epoch {}".format(epoch))
    plt.imshow(np.transpose(vutils.make_grid(x_tilde_val.squeeze().cpu(), padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(args.stdout, 'val_gen_images_ep_{}.png'.format(epoch)))
    plt.close()

    plt.figure()
    plt.axis("off")
    plt.title("Test Generated Images Epoch {}".format(epoch))
    plt.imshow(np.transpose(vutils.make_grid(x_tilde_test.squeeze().cpu(), padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(args.stdout, 'test_gen_images_ep_{}.png'.format(epoch)))
    plt.close()


def prepare_random_batch(tr_loader, val_loader, te_loader, args):
    n = args.num_save_image
    x1_tr, x2_tr, y_tr, x1_val, x2_val, y_val, x1_te, x2_te, y_te = [], [], [], [], [], [], [], [], []
    itr_tr, itr_val, itr_te = iter(tr_loader), iter(val_loader), iter(te_loader)
    for i in range(n):
        tr_sample = next(itr_tr)
        x1_b_tr, x2_b_tr, y_b_tr = tr_sample[0][0].to(args.device), tr_sample[0][1].to(args.device), tr_sample[1].to(
            args.device)
        x1_tr.append(x1_b_tr)
        x2_tr.append(x2_b_tr)
        y_tr.append(y_b_tr)

        val_sample = next(itr_val)
        x1_b_val, x2_b_val, y_b_val = val_sample[0][0].to(args.device), val_sample[0][1].to(args.device), val_sample[1] \
            .to(args.device)
        x1_val.append(x1_b_val)
        x2_val.append(x2_b_val)
        y_val.append(y_b_val)

        te_sample = next(itr_te)
        x1_b_te, x2_b_te, y_b_te = te_sample[0][0].to(args.device), te_sample[0][1].to(args.device), te_sample[1].to(
            args.device)
        x1_te.append(x1_b_te)
        x2_te.append(x2_b_te)
        y_te.append(y_b_te)

    x1_tr, x2_tr, y_tr = (torch.stack(x1_tr).squeeze())[:n], (torch.stack(x2_tr).squeeze())[:n], (torch.stack(
        y_tr).squeeze())[:n]
    x1_val, x2_val, y_val = (torch.stack(x1_val).squeeze())[:n], (torch.stack(x2_val).squeeze())[:n], (torch.stack(
        y_val).squeeze())[:n]
    x1_te, x2_te, y_te = (torch.stack(x1_te).squeeze())[:n], (torch.stack(x2_te).squeeze())[:n], (torch.stack(
        y_te).squeeze())[:n]

    batch_tr, batch_val, batch_te = ((x1_tr, x2_tr), y_tr), ((x1_val, x2_val), y_val), ((x1_te, x2_te), y_te)
    logging.warning('*****************************************************')
    logging.warning('Sampled batches:')
    logging.warning('Train:')
    logging.warning(y_tr)
    logging.warning('Val:')
    logging.warning(y_val)
    logging.warning('Test:')
    logging.warning(y_te)
    logging.warning('*****************************************************')
    return batch_tr, batch_val, batch_te


def add_losses_to_writer(writer, losses, epoch=None, iteration=None, on_epoch=True):
    loss_names = losses.keys()
    if on_epoch:
        for name in loss_names:
            writer.add_scalar('Train/{}_Epoch'.format(name), losses[name], epoch)
    else:
        for name in loss_names:
            writer.add_scalar('Train/{}_Iter'.format(name), losses[name], iteration)
    return writer


def prepare_logging_string(losses, epoch, num_epochs, itr, num_iteration):
    log = 'Epoch: {}/{} \t Iter: {}/{}'.format(epoch, num_epochs, itr, num_iteration)
    for k, v in losses.items():
        log += ' \t {}: {:.4f}'.format(k, v)
    return log


def train_next_step_classifier(model, datamodule, args):
    train_dataloader, val_dataloader, test_dataloader = datamodule.train_dataloader(), datamodule.val_dataloader(), \
                                                        datamodule.test_dataloader()
    checkpoint = {'model': None, 'val_loss': 1e10, 'val_acc': None, 'val_auc': None, 'test_loss': None,
                  'test_acc': None, 'test_auc': None, 'iter': None}
    epoch, iteration = 0, 0
    model.train()
    if not args.debug:
        for epoch in range(1, args.num_epochs + 1):
            epoch_loss = 0.
            for itr, batch in enumerate(train_dataloader, 0):
                # if itr == 1:
                #     break
                x1, y = batch[0][0].to(args.device), batch[1].to(args.device)
                y_pred = model(x1)
                loss = args.class_loss(y_pred, y)
                args.optimizer_c.zero_grad()
                loss.backward()
                args.optimizer_c.step()
                args.writer.add_scalar('Train/Loss_Iter', loss, iteration)
                epoch_loss += loss
                iteration += 1
                if (itr + 1) % args.num_test_iters == 0:
                    eval_val = test_next_step_classifier(model, val_dataloader, args)
                    eval_test = test_next_step_classifier(model, test_dataloader, args)
                    args.writer.add_scalar('Val/Loss_Iter', eval_val['loss'], iteration)
                    args.writer.add_scalar('Val/Acc_Iter', eval_val['acc'], iteration)
                    args.writer.add_scalar('Val/AUC_Iter', eval_val['auc'], iteration)
                    args.writer.add_scalar('Test/Loss_Iter', eval_test['loss'], iteration)
                    args.writer.add_scalar('Test/Acc_Iter', eval_test['acc'], iteration)
                    args.writer.add_scalar('Test/AUC_Iter', eval_test['auc'], iteration)
                    logging.warning(
                        'Epoch: {}/{} \t Iter: {}/{} \t Loss: {:.4f} \t Val_Acc: {:.4f} \t Val_AUC: {:.4f} \t'
                        ' Test_Acc: {:.4f} \t Test_AUC: {:.4f}'.
                            format(epoch, args.num_epochs, itr + 1, len(train_dataloader), loss.item(), eval_val['acc'],
                                   eval_val['auc'], eval_test['acc'], eval_test['auc']))
                    if eval_val['loss'] < checkpoint['val_loss']:
                        checkpoint['model'] = copy.deepcopy(model.state_dict())
                        checkpoint['val_acc'], checkpoint['val_loss'], checkpoint['val_auc'] = eval_val['acc'], \
                                                                                               eval_val['loss'], \
                                                                                               eval_val['auc']
                        checkpoint['test_acc'], checkpoint['test_loss'], checkpoint['test_auc'] = eval_test['acc'], \
                                                                                                  eval_test['loss'], \
                                                                                                  eval_test['auc']
                        checkpoint['iter'] = iteration
                else:
                    logging.warning('Epoch: {}/{} \t Iter: {}/{} \t Loss: {:.4f}'.
                                    format(epoch, args.num_epochs, itr + 1, len(train_dataloader), loss.item()))

            # Testing at the end of epoch.
            epoch_loss = epoch_loss / len(train_dataloader)
            logging.warning('Evaluating Network at the end of epoch #{}:'.format(epoch))
            eval_val = test_next_step_classifier(model, val_dataloader, args)
            eval_test = test_next_step_classifier(model, test_dataloader, args)
            args.writer.add_scalar('Train/Loss_Epoch', epoch_loss, epoch)
            args.writer.add_scalar('Val/Loss_Epoch', eval_val['loss'], epoch)
            args.writer.add_scalar('Val/Acc_Epoch', eval_val['acc'], epoch)
            args.writer.add_scalar('Val/AUC_Epoch', eval_val['auc'], epoch)
            args.writer.add_scalar('Test/Loss_Epoch', eval_test['loss'], epoch)
            args.writer.add_scalar('Test/Acc_Epoch', eval_test['acc'], epoch)
            args.writer.add_scalar('Test/AUC_Epoch', eval_test['auc'], epoch)
            logging.warning('Epoch: {}/{} \t Epoch_Loss: {:.4f} \t Val_Acc: {:.4f} \t Val_AUC: {:.4f} \t '
                            'Test_Acc: {:.4f} \t Test_AUC: {:.4f}'.
                            format(epoch, args.num_epochs, epoch_loss.item(), eval_val['acc'], eval_val['auc'],
                                   eval_test['acc'], eval_test['auc']))
            if eval_val['loss'] < checkpoint['val_loss']:
                checkpoint['model'] = copy.deepcopy(model.state_dict())
                checkpoint['val_acc'], checkpoint['val_loss'], checkpoint['val_auc'] = eval_val['acc'], eval_val[
                    'loss'], eval_val['auc']
                checkpoint['test_acc'], checkpoint['test_loss'], checkpoint['test_auc'] = eval_test['acc'], eval_test[
                    'loss'], eval_test['auc']
                checkpoint['iter'] = iteration

        return checkpoint

    else:
        model.train()
        train_batch = next(iter(train_dataloader))
        x, y = train_batch[0].to(args.device), train_batch[1].to(args.device)
        for epoch in range(args.num_epochs):
            model.train()
            y_pred = model(x)
            loss = args.criterion(y_pred, y)
            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()
            epoch_loss = loss
            eval_val = test(model, val_dataloader, args)
            eval_test = test(model, test_dataloader, args)
            args.writer.add_scalar('Train/Loss_Epoch', epoch_loss.item(), epoch)
            args.writer.add_scalar('Val/Loss_Epoch', eval_val['loss'], epoch)
            args.writer.add_scalar('Val/Acc_Epoch', eval_val['acc'], epoch)
            args.writer.add_scalar('Val/AUC_Epoch', eval_val['auc'], epoch)
            args.writer.add_scalar('Test/Loss_Epoch', eval_test['loss'], epoch)
            args.writer.add_scalar('Test/Acc_Epoch', eval_test['acc'], epoch)
            args.writer.add_scalar('Test/AUC_Epcoh', eval_test['auc'], epoch)
            logging.warning('Epoch: {}/{} \t Epoch_Loss: {:.4f} \t Val_Acc: {:.4f} \t Val_AUC: {:.4f} \t '
                            'Test_Acc: {:.4f} \t Test_AUC: {:.4f}'.
                            format(epoch, args.num_epochs, epoch_loss.item(), eval_val['acc'], eval_val['auc'],
                                   eval_test['acc'], eval_test['auc']))
            if eval_val['loss'] < checkpoint['val_loss']:
                checkpoint['model'] = copy.deepcopy(model.state_dict())
                checkpoint['val_acc'], checkpoint['val_loss'], checkpoint['val_auc'] = eval_val['acc'], eval_val[
                    'loss'], eval_val['auc']
                checkpoint['test_acc'], checkpoint['test_loss'], checkpoint['test_auc'] = eval_test['acc'], eval_test[
                    'loss'], eval_test['auc']
        return checkpoint


def test_next_step_classifier(model, test_dataloader, args):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_preds = []
    y_true = []
    with torch.no_grad():
        for data in test_dataloader:
            x, y = data[0][0].to(args.device), data[1].to(args.device)
            y_pred = model(x)
            y_preds.append(y_pred)
            y_true.append(y)
            loss = args.class_loss(y_pred, y)
            total_loss += loss
            _, label_pred = torch.max(y_pred, dim=1)
            total += y.shape[0]
            correct += (label_pred == y).sum().item()
    total_loss = total_loss / total
    accuracy = correct / total
    y_preds = torch.softmax(torch.stack(y_preds, dim=0).squeeze(), dim=1)
    y_true = torch.stack(y_true).squeeze()
    if not args.debug:
        if args.binary:
            auc = roc_auc_score(y_true.cpu().numpy(), (y_preds[:, 1].squeeze()).cpu().numpy())
        else:
            auc = roc_auc_score(y_true.cpu().numpy(), y_preds.cpu().numpy(), multi_class='ovo')
    else:
        auc = 0.
    model.train()
    return {'loss': total_loss, 'acc': accuracy, 'auc': auc}
