
Learn more or give us feedback
import logging
import copy
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

import model.learner as Learner

logger = logging.getLogger("experiment")


class MetaLearingClassification(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config, treatment):

        super(MetaLearingClassification, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step

        if treatment == "Neuromodulation":
            neuromodulation = True
        else:
            neuromodulation = False

        self.net = Learner.Learner(config, neuromodulation)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.meta_iteration = 0
        self.inputNM = True
        self.nodeNM = False
        self.layers_to_fix = []

    def reset_classifer(self, class_to_reset):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def reset_layer(self, layer_to_reset):
        if layer_to_reset % 2 == 0:
            weight = self.net.parameters()[layer_to_reset]#-2]
            torch.nn.init.kaiming_normal_(weight)
        else:
            bias = self.net.parameters()[layer_to_reset]
            bias.data = torch.ones(bias.data.size())

    def add_patch_to_images(self, images, task_num):
        boxSize = 8
        if task_num == 1:
            try:
                images[:,:,:boxSize+1,:boxSize+1] = torch.min(images)
            except:
                images[:,:boxSize+1,:boxSize+1] = torch.min(images)
        elif task_num == 2:
            images[:,:,-(boxSize+1):,-(boxSize+1):] = torch.min(images)
        elif task_num == 3:
            images[:,:,:boxSize+1, -(boxSize+1):] = torch.min(images)
        elif task_num == 4:
            images[:,:,-(boxSize+1):, :boxSize+1] = torch.min(images)
        return images

    def shuffle_labels(self, targets, batch=False):
        if batch:
            new_target = (targets[0]+2)%1000
            for t in range(len(targets)):
                targets[t] = new_target

            return(targets)
        
        else:
            new_target = (targets+2)%1000
            return(new_target)

    def sample_training_data(self, iterators, it2, steps=2, reset=True):

        # Sample data for inner and meta updates

        x_traj = []
        y_traj = []
        x_rand = []
        y_rand = []

        counter = 0

        class_cur = 0
        class_to_reset = 0
        for it1 in iterators:
            for img, data in it1:
                class_to_reset = data[0].item()
                if reset:
                    #next
                    # Resetting weights corresponding to classes in the inner updates; this prevents
                    # the learner from memorizing the data (which would kill the gradients due to inner updates)
                    self.reset_classifer(class_to_reset)
                    #self.bn_reset_counter = 0
                    #layers_to_reset = [18,19,20,21,22,23,24,25,26,27,28,29]
                    #for layer in layers_to_reset:
                    #    self.reset_layer(layer)

                #self.net.cuda()
                counter += 1
                x_traj.append(img)
                y_traj.append(data)
                if counter % int(steps / len(iterators)) == 0:
                    class_cur += 1
                    break

        # To handle a corner case; nothing interesting happening here
        if len(x_traj) < steps:
            it1 = iterators[-1]
            for img, data in it1:
                counter += 1
                x_traj.append(img)
                y_traj.append(data)
                if counter % int(steps % len(iterators)) == 0:
                    break

        # Sampling the random batch of data
        counter = 0
        for img, data in it2:
            if counter == 1:
                break
            x_rand.append(img)
            y_rand.append(data)
            counter += 1

        class_cur = 0
        counter = 0
        x_rand_temp = []
        y_rand_temp = []
        for it1 in iterators:
            for img, data in it1:
                counter += 1
                x_rand_temp.append(img)
                y_rand_temp.append(data)
                if counter % int(steps / len(iterators)) == 0:
                    class_cur += 1
                    break

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)
        x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(
            y_rand)

        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)

        return x_traj, y_traj, x_rand, y_rand

    def inner_update(self, x, fast_weights, y, bn_training):

        logits = self.net(x, fast_weights, bn_training=bn_training)
        loss = F.cross_entropy(logits, y)

        if fast_weights is None:
            fast_weights = self.net.parameters()

        grad = torch.autograd.grad(loss, fast_weights, allow_unused=False)

        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
            params_new.learn = params_old.learn

        return fast_weights

    def meta_loss(self, x, fast_weights, y, bn_training):

        logits = self.net(x, fast_weights, bn_training=bn_training)
        loss_q = F.cross_entropy(logits, y)
        return loss_q, logits

    def eval_accuracy(self, logits, y):
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()
        return correct

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        """
        :param x_traj:   Input data of sampled trajectory
        :param y_traj:   Ground truth of the sampled trajectory
        :param x_rand:   Input data of the random batch of data
        :param y_rand:   Ground truth of the random batch of data
        :return:
        """

        black_square = False
        if black_square:

            #coin_flip = np.random.randn()
            #if coin_flip > 0:
            x_traj_bs = self.add_patch_to_images(copy.deepcopy(x_traj), task_num=1)
            y_traj_bs = self.shuffle_labels(copy.deepcopy(y_traj), batch=True)
            
            # randomly add a black patch to the validation images
            for i in range(len(x_rand[0])):
                coin_flip = np.random.randn()
                if coin_flip > 0:
                    x_rand[0][i] = self.add_patch_to_images(x_rand[0][i], task_num=1)
                    y_rand[0][i] = self.shuffle_labels(y_rand[0][i], batch=False)

                    #plt.imshow(x_rand[0][i][0,:,:])
                    #plt.show()

        fast_weights = self.inner_update(x_traj[0], None, y_traj[0], False)
        
        for k in range(1, self.update_step):
            # Doing inner updates using fast weights
            fast_weights = self.inner_update(x_traj[k], fast_weights, y_traj[k], False)
        
        meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], False)
     
        with torch.no_grad():
            pred_q = F.softmax(logits, dim=1).argmax(dim=1)
            classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy

        # Taking the meta gradient step
    
        self.net.zero_grad()

        NM_reset = False
    
        if NM_reset:

            layers_to_reset = list(range(14, 28))
            grads = torch.autograd.grad(meta_loss, self.net.parameters())
        
            for idx in range(len(self.net.parameters())):
                if idx in layers_to_reset:
                    self.net.parameters()[idx].grad = None
                else:
                    self.net.parameters()[idx].grad = grads[idx]
        else:
            meta_loss.backward()

        self.optimizer.step()
        
        classification_accuracy /= len(x_rand[0])
        
        self.meta_iteration += 1

        return classification_accuracy, meta_loss


class MetaLearnerRegression(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):
        """
        :param args:
        """
        super(MetaLearnerRegression, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step

        self.net = Learner.Learner(config)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.meta_optim = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [1500, 2500, 3500], 0.1)

    def forward(self, x_traj, y_traj, x_rand, y_rand):

        losses_q = [0 for _ in range(len(x_traj) + 1)]

        for i in range(1):
            logits = self.net(x_traj[0], vars=None, bn_training=False)
            logits_select = []
            for no, val in enumerate(y_traj[0, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss = F.mse_loss(logits, y_traj[0, :, 0].unsqueeze(1))
            grad = torch.autograd.grad(loss, self.net.parameters())

            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, self.net.parameters())))
            for params_old, params_new in zip(self.net.parameters(), fast_weights):
                params_new.learn = params_old.learn

            with torch.no_grad():

                logits = self.net(x_rand[0], vars=None, bn_training=False)

                logits_select = []
                for no, val in enumerate(y_rand[0, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                losses_q[0] += loss_q

            for k in range(1, len(x_traj)):
                logits = self.net(x_traj[k], fast_weights, bn_training=False)

                logits_select = []
                for no, val in enumerate(y_traj[k, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)

                loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

                for params_old, params_new in zip(self.net.parameters(), fast_weights):
                    params_new.learn = params_old.learn

                logits_q = self.net(x_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), :], fast_weights,
                                    bn_training=False)

                logits_select = []
                for no, val in enumerate(y_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), 1].long()):
                    logits_select.append(logits_q[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), 0].unsqueeze(1))

                losses_q[k + 1] += loss_q

        self.optimizer.zero_grad()

        loss_q = losses_q[k + 1]
        loss_q.backward()
        self.optimizer.step()

        return losses_q


def main():
    pass


if __name__ == '__main__':
    main()
