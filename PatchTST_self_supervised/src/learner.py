
from typing import List
import torch
from torch.optim import Adam
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .basics import *
from .callback.core import * 
from .callback.tracking import * 
from .callback.scheduler import *
from .callback.distributed import *
from .utils import *
from pathlib import Path
from tqdm import tqdm

import numpy as np

from sklearn.base import BaseEstimator
from unittest.mock import patch


class Learner(GetAttr):

    def __init__(self, dls, model, 
                        loss_func=None, 
                        lr=1e-3, 
                        cbs=None, 
                        metrics=None, 
                        opt_func=Adam,
                        **kwargs):
                
        self.model, self.dls, self.loss_func, self.lr = model, dls, loss_func, lr
        self.opt_func = opt_func
        #self.opt = self.opt_func(self.model.parameters(), self.lr) 
        self.set_opt()
        
        self.metrics = metrics
        self.n_inp  = 2
        # self.n_inp = self.dls.train.dataset.n_inp if self.dls else 0
        # Initialize callbacks                 
        if cbs and not isinstance(cbs, List): cbs = [cbs]    
        self.initialize_callbacks(cbs)        
        # Indicator of running lr_finder
        self.run_finder = False

    def set_opt(self):
        if self.model:
            self.opt = self.opt_func(self.model.parameters(), self.lr) 
        else: self.opt = None


    def default_callback(self):
        "get a set of default callbacks"
        default_cbs = [ SetupLearnerCB(), TrackTimerCB(), 
                        TrackTrainingCB(train_metrics=False, valid_metrics=True)]                  
        return default_cbs
    

    def initialize_callbacks(self, cbs):        
        default_cbs = self.default_callback()       
        self.cbs = update_callbacks(cbs, default_cbs) if cbs else default_cbs        
        # add print CB
        self.cbs += [PrintResultsCB()]        
        for cb in self.cbs: cb.learner = self     
        self('init_cb')       


    def add_callback(self, cb):                
        if not cb: return
        cb.learner = self
        self.cbs = update_callback(cb, self.cbs)           

    def add_callbacks(self, cbs):        
        if not isinstance(cbs, list):  cbs = [cbs]
        for cb in cbs: self.add_callback(cb)

    def remove_callback(self, cb): 
        cb.learn = None
        self.cbs, removed_cb = remove_callback(cb, self.cbs)
        return removed_cb
    
    def remove_callbacks(self, cb_list):
        for cb in cb_list: self.remove_callback(cb)


    def fit(self, n_epochs, lr=None, cbs=None, do_valid=True):
        " fit the model "
        self.n_epochs = n_epochs
        if not self.dls.valid: do_valid = False
        if cbs: self.add_callbacks(cbs)
        if lr: self.opt = self.opt_func(self.model.parameters(), lr) 

        self('before_fit')
        try:
            for self.epoch in range(n_epochs):            
                self('before_epoch')                     
                self.one_epoch(train=True)            
                # if self.dls.valid:                    
                if do_valid: self.one_epoch(train=False)                
                self('after_epoch')        
        except KeyboardInterrupt: pass 
        self('after_fit')


    def fit_one_cycle(self, n_epochs, lr_max=None, pct_start=0.3):
        self.n_epochs = n_epochs        
        self.lr_max = lr_max if lr_max else self.lr
        cb = OneCycleLR(lr_max=self.lr_max, pct_start=pct_start)
        self.fit(self.n_epochs, cbs=cb)                
         
         
    def one_epoch(self, train):                           
        self.epoch_train() if train else self.epoch_validate()        

    def epoch_train(self):
        self('before_epoch_train')
        self.model.train()                
        self.dl = self.dls.train
        self.all_batches('train')
        self('after_epoch_train')
    
    def epoch_validate(self, dl=None):
        self('before_epoch_valid')
        # model at evaluation mode  
        self.model.eval()                
        self.dl = dl if dl else self.dls.valid
        if self.dl:        
            with torch.no_grad(): self.all_batches('valid')
        self('after_epoch_valid')


    def all_batches(self, type_):
        # for self.num,self.batch in enumerate(progress_bar(dl, leave=False)):        
        for num, batch in enumerate(self.dl):            
            self.iter, self.batch = num, batch            
            if type_ == 'train': self.batch_train()
            elif type_ == 'valid': self.batch_validate()
            elif type_ == 'predict': self.batch_predict()             
            elif type_ == 'test': self.batch_test()

    def batch_train(self):
        self('before_batch_train')
        self._do_batch_train()
        self('after_batch_train')  

    def batch_validate(self):
        self('before_batch_valid')
        self._do_batch_validate()
        self('after_batch_valid')  
    
    def batch_predict(self):
        self('before_batch_predict')
        self._do_batch_predict()
        self('after_batch_predict') 

    def batch_test(self):
        self('before_batch_test')
        self._do_batch_test()
        self('after_batch_test') 
        
    def _do_batch_train(self):        
        # forward + get loss + backward + optimize          
        self.pred, self.loss = self.train_step(self.batch)                                      
        # zero the parameter gradients
        self.opt.zero_grad()                 
        # gradient
        self.loss.backward()
        # update weights
        self.opt.step() 

    def train_step(self, batch):
        # get the inputs
        self.xb, self.yb = batch
        # forward
        pred = self.model_forward()
        # compute loss
        loss = self.loss_func(pred, self.yb)
        return pred, loss

    def model_forward(self):
        self('before_forward')
        self.pred = self.model(self.xb)
        self('after_forward')
        return self.pred

    def _do_batch_validate(self):       
        # forward + calculate loss
        self.pred, self.loss = self.valid_step(self.batch)     

    def valid_step(self, batch):
        # get the inputs
        self.xb, self.yb = batch
        # forward
        pred = self.model_forward()
        # compute loss
        loss = self.loss_func(pred, self.yb)
        return pred, loss                                     


    def _do_batch_predict(self):   
        self.pred = self.predict_step(self.batch)     
           
    def predict_step(self, batch):
        # get the inputs
        self.xb, self.yb = batch
        # forward
        pred = self.model_forward()
        return pred 
    
    def _do_batch_test(self):   
        self.pred, self.yb = self.test_step(self.batch)     
           
    def test_step(self, batch):
        # get the inputs
        self.xb, self.yb = batch
        # forward
        pred = self.model_forward()
        return pred, self.yb


    def _predict(self, dl=None):
        # self('before_validate')
        self('before_predict')
        if dl is None: return
        self.dl = dl
        self.n_inp = dl.dataset.n_inp                
        self.model.eval()        #  model at evaluation mode  
        with torch.no_grad(): self.all_batches('predict')        
        self('after_predict')


    def predict(self, test_data, weight_path=None, Dataset=None, Dataloader=None, batch_size=None):
        """_summary_
        Args:
            test_data can be a tensor, numpy array, dataset or dataloader
        Returns:
            _type_: _description_
        """                
        if weight_path is not None: self.load(weight_path)
        cb = GetPredictionsCB()
        self.add_callback(cb)                    
        test_dl = self._prepare_data(test_data, Dataset, Dataloader, batch_size)
        self._predict(test_dl)        
        self.preds = cb.preds
        return to_numpy(self.preds) 
   
    
    def test(self, dl, weight_path=None, scores=None):
        """_summary_
        Args:
            test_data can be a tensor, numpy array, dataset or dataloader
        Returns:
            _type_: _description_
        """          
        if dl is None: return
        else: self.dl = dl
        if weight_path is not None: self.load(weight_path)
        cb = GetTestCB()
        self.add_callback(cb)
        self('before_test')
        self.model.eval()
        with torch.no_grad(): self.all_batches('test')
        self('after_test')   
        self.preds, self.targets = to_numpy([cb.preds, cb.targets])
        # calculate scores
        if scores: 
            s_vals = [score(cb.targets, cb.preds).to('cpu').numpy() for score in list(scores)]
            return self.preds, self.targets, s_vals
        else: return self.preds, self.targets


    def _prepare_data(self, test_data, Dataset=None, Dataloader=None, batch_size=None):
        if test_data is None: return test_data
        if Dataset and Dataloader:
            test_dset = Dataset(test_data)
            if not batch_size: batch_size=16
            test_dl = Dataloader(test_dset, batch_size)        
        else:            
            if self.dls: 
                # add test_data to the dataloader defined in the dls.train
                test_dl = self.dls.add_dl(test_data, batch_size=batch_size)  
            else: test_dl = test_data       # assume test_data is already a form of dataloader
        return test_dl
   
    
    def get_layer_output(self, inp, layers=None, unwrap=False):
        """
        Args:
            inp: can be numpy array, torch tensor or dataloader
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        if isinstance(inp, np.ndarray): inp = torch.Tensor(inp).to(device)
        if isinstance(inp, torch.Tensor): inp = inp.to(device)
        
        return get_layer_output(inp, model=self.model, layers=layers, unwrap=unwrap)
    

    def fine_tune(self, n_epochs, base_lr=None, freeze_epochs=1, pct_start=0.3):
        """
        fintune the pretrained model. First the entire model is freezed, only head is trained
        up to a freeze_epochs number. Then the model is unfreezed and the entire model is trained
        """
        assert (n_epochs>0)|(freeze_epochs>0), "Either n_epochs or freeze_epochs has to be > 0"
        if not base_lr: base_lr = self.lr
        # Finetune the head of freeze_epochs > 0:
        if freeze_epochs > 0:
            print('Finetune the head')
            self.freeze()
            self.fit_one_cycle(freeze_epochs, lr_max=base_lr, pct_start=pct_start)
        
        # Finetune the entire network if n_epochs > 0
        if n_epochs > 0:
            print('Finetune the entire network')        
            self.unfreeze()
            self.fit_one_cycle(n_epochs, lr_max=base_lr/2, pct_start=pct_start)
    

    def linear_probe(self, n_epochs, base_lr=None, pct_start=0.3):
        """
        linear probing the pretrained model. The model is freeze except the head during finetuning
        """
        assert (n_epochs>0), "n_epochs has to be > 0"
        if not base_lr: base_lr = self.lr
        print('Finetune the head')
        self.freeze()
        self.fit_one_cycle(n_epochs, lr_max=base_lr, pct_start=pct_start)
    

    def lr_finder(self, start_lr=1e-7, end_lr=10, num_iter=100, step_mode='exp', show_plot=True, suggestion='valley'):                
        """
        find the learning rate
        """
        n_epochs = num_iter//len(self.dls.train) + 1
        # indicator of lr_finder method is applied
        self.run_finder = True
        # add LRFinderCB to callback list and will remove later
        cb = LRFinderCB(start_lr, end_lr, num_iter, step_mode, suggestion=suggestion)                
        # fit           
        self.fit(n_epochs=n_epochs, cbs=cb, do_valid=False)        
        # should remove LRFinderCB callback after fitting                
        self.remove_callback(cb)        
        self.run_finder = False        
        if show_plot: cb.plot_lr_find()
        if suggestion: return cb.suggested_lr  
        
        

    def freeze(self):
        """ 
        freeze the model head
        require the model to have head attribute
        """
        if hasattr(get_model(self.model), 'head'): 
            # print('model head is available')
            for param in get_model(self.model).parameters(): param.requires_grad = False        
            for param in get_model(self.model).head.parameters(): param.requires_grad = True
            # print('model is frozen except the head')
            
            
    def unfreeze(self):
        for param in get_model(self.model).parameters(): param.requires_grad = True        


    def __call__(self, name):        
        for cb in self.cbs: 
            attr = getattr(cb, name)
            if attr is not None: attr()
          

    def save(self, fname, path, **kwargs):
        """
        Save model and optimizer state (if `with_opt`) to `self.path/file`
        """
        fname = join_path_file(fname, path, ext='.pth')        
        save_model(fname, self.model, getattr(self,'opt',None), **kwargs)
        return fname


    def load(self, fname, with_opt=False, device='cuda', strict=True, **kwargs):
        """
        load the model
        """
        if not torch.cuda.is_available():
            device = "cpu"
        load_model(fname, self.model, self.opt, with_opt, device=device, strict=strict)


    def get_params(self, deep=True, **kwargs):
        params = BaseEstimator.get_params(self, deep=deep, **kwargs)
        return params

    def _get_param_names(self):
        return (k for k in self.__dict__ if not k.endswith('_'))


    def set_params(self, **kwargs):
        params = {}
        for key, val in kwargs.items():
            params[key] = val
        BaseEstimator.set_params(self, **params)

    def to_distributed(self,
                       sync_bn=True,  # Whether to replace all batch norm with `nn.SyncBatchNorm`
                       **kwargs
                       ):
        local_rank = int(os.environ.get('LOCAL_RANK'))
        world_size = int(os.environ.get('WORLD_SIZE'))
        rank = int(os.environ.get('RANK'))
        print('Process {} (out of {})'.format(
            rank, torch.distributed.get_world_size()))

        self.add_callback(DistributedTrainer(local_rank=local_rank, world_size=world_size, sync_bn=sync_bn, **kwargs))

        return self


def save_model(path, model, opt, with_opt=True, pickle_protocol=2):
    "Save `model` to `file` along with `opt` (if available, and if `with_opt`)"
    if opt is None: with_opt=False
    state = get_model(model).state_dict()
    if with_opt: state = {'model': state, 'opt':opt.state_dict()}
    torch.save(state, path, pickle_protocol=pickle_protocol)


def load_model(path, model, opt=None, with_opt=False, device='cpu', strict=True):
    " load the saved model "
    state = torch.load(path, map_location=device)
    if not opt: with_opt=False
    model_state = state['model'] if with_opt else state
    get_model(model).load_state_dict(model_state, strict=strict)
    if with_opt: opt.load_state_dict(state['opt'])
    model = model.to(device)
      

def join_path_file(file, path, ext=''):
    "Return `path/file` if file is a string or a `Path`, file otherwise"
    if not isinstance(file, (str, Path)): return file
    if not isinstance(path, Path): path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path/f'{file}{ext}'


def get_model(model):
    "Return the model maybe wrapped inside `model`."    
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model


def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    # state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)
    #print('new_state_dict',new_state_dict)
    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items():        
        if exclude_head and 'head' in name: continue
        if name in new_state_dict:            
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape: param.copy_(input_param)
            else: unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0: raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model


def update_callback(cb, list_cbs):
    for cb_ in list_cbs:
        if type(cb_) ==  type(cb): list_cbs.remove(cb_)
    list_cbs += [cb]
    return list_cbs

def update_callbacks(list_cbs, default_cbs):
    for cb in list_cbs: default_cbs = update_callback(cb, default_cbs)
    return default_cbs

def remove_callback(cb, list_cbs):
    for cb_ in list_cbs:
        if type(cb_) ==  type(cb):             
            list_cbs.remove(cb_)
            break
    return list_cbs, cb_


def get_layer_output(inp, model, layers=None, unwrap=False):
    """
    layers is a list of module names
    """
    orig_model = model
    
    if unwrap: model = unwrap_model(model)
    if not layers: layers = list(dict(model.named_children()).keys())
    if not isinstance(layers, list): layers = [layers]

    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()
        return hook

    # register forward hooks on the layers of choice    
    h_list = [getattr(model, layer).register_forward_hook(getActivation(layer)) for layer in layers]
    
    model.eval()
    out = orig_model(inp)    
    for h in h_list: h.remove()
    return activation
