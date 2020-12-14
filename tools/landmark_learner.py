import  os
import  torch
import  tqdm
import  numpy as np
from torch.utils.data import  Dataset, DataLoader
from sklearn.metrics import classification_report,accuracy_score
from torch.optim.optimizer import Optimizer

from fer_pytorch.utils.logger import TxtLogger
from fer_pytorch.utils.meter import AverageMeter

class LandmarkLearner():
    def __init__(self, model : torch.nn.Module,
                 loss_fn : torch.nn.Module,
                 optimizer: Optimizer,
                 scheduler,
                 logger : TxtLogger,
                 save_dir : str,
                 log_steps = 100,
                 device_ids = [0,1],
                 gradient_accum_steps = 1,
                 max_grad_norm = 1.0,
                 batch_to_model_inputs_fn = None,
                 early_stop_n = 4,
                 ):
        self.model  = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.logger = logger
        self.device_ids =device_ids
        self.gradient_accum_steps = gradient_accum_steps
        self.max_grad_norm = max_grad_norm
        self.batch_to_model_inputs_fn  = batch_to_model_inputs_fn
        self.early_stop_n = early_stop_n
        self.global_step = 0

    def step(self,step_n,  batch_data : dict):
        preds = self.model(batch_data['imgs'])
        labels = batch_data['labels']
        norm_dis = batch_data['norm_dis']
        label = labels.squeeze()
        loss = self.loss_fn(preds, label)
        loss = loss.mean()
        if self.gradient_accum_steps > 1:
            loss = loss / self.gradient_accum_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (step_n + 1) % self.gradient_accum_steps == 0:
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            self.global_step += 1

        nme = self._NME(preds, labels, norm_dis)
        return  loss, nme

    def _NME(self,preds, labels,norm_dis):
        norm_dis = norm_dis.squeeze()
        diff = torch.abs(preds - labels).mean(axis=-1)
        diff = diff / norm_dis
        nme  = diff.mean()
        return  nme

    def _batch_trans(self, batch):
        batch = tuple(t.to(self.device_ids[0]) for t in batch if isinstance(t, torch.Tensor))
        if self.batch_to_model_inputs_fn is None:
            batch_data = {
                'imgs': batch[0],
                'labels': batch[1],
                'norm_dis':batch[2]
            }
        else:
            batch_data = self.batch_to_model_inputs_fn(batch)
        return  batch_data

    def val(self, val_dataloader : DataLoader):
        eval_loss = 0.0
        eval_nme  = 0.0
        self.model.eval()
        for batch in tqdm.tqdm(val_dataloader):
            with torch.no_grad():
                batch_data = self._batch_trans(batch)
                preds = self.model( batch_data['imgs'])
                labels = batch_data["labels"].squeeze()
                norm_dis = batch_data["norm_dis"]
                loss = self.loss_fn(preds,labels )
                nme = self._NME(preds, labels, norm_dis)
                eval_loss += loss.mean().item()
                eval_nme += nme.mean().item()

        eval_loss = eval_loss / len(val_dataloader)
        eval_nme  = eval_nme  / len(val_dataloader)
        self.logger.write("steps: {} ,mean eval loss : {:.5f} eval NME:{:.6f} ". \
                                      format(self.global_step, eval_loss,eval_nme))
        return {'nme': eval_nme}

    def train(self, train_dataloader : DataLoader,
              val_dataloader : DataLoader,
              epoches = 100):
        best_nme = 1e9
        early_n = 0
        for epo in range(epoches):
            step_n = 0
            train_avg_loss = AverageMeter()
            train_avg_acc = AverageMeter()
            data_iter = tqdm.tqdm(train_dataloader)
            for batch in data_iter:
                self.model.train()
                batch_data = self._batch_trans(batch)
                train_loss, acc = self.step(step_n, batch_data)
                train_avg_loss.update(train_loss.item(),1)
                train_avg_acc.update(acc,1)
                status = '[{0}] lr= {1:.6f} loss= {2:.5f} avg_loss= {3:.5f} avg_nme={4:.5f} '.format(
                    epo + 1, self.scheduler.get_lr()[0],
                    train_loss.item(), train_avg_loss.avg, train_avg_acc.avg )
                #if step_n%self.log_steps ==0:
                #    print(status)
                data_iter.set_description(status)
                step_n +=1

            ##self.scheduler.step() ## we update every step instead
            if True:
                ## val
                m = self.val(val_dataloader)
                nme = m['nme']
                if best_nme > nme:
                    early_n = 0
                    best_nme = nme
                    model_path = os.path.join(self.save_dir, 'best.pth')
                    torch.save(self.model.state_dict(), model_path)
                else:
                    early_n += 1
                self.logger.write("steps: {} ,mean NME : {:.5f} , best NME: {:.5f}". \
                                      format(self.global_step, nme, best_nme))
                self.logger.write(str(m))
                self.logger.write("=="*50)

                if early_n > self.early_stop_n:
                    print('early stopped!')
                    return best_nme
        return  best_nme








