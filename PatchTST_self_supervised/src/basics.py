
import torch

import collections
from collections import OrderedDict

class GetAttr:

    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _default='default'
    def _component_attr_filter(self,k):
        if k.startswith('__') or k in ('_xtra',self._default): return False
        xtra = getattr(self,'_xtra',None)
        return xtra is None or k in xtra

    def _dir(self): 
        return [k for k in dir(getattr(self,self._default)) if self._component_attr_filter(k)]

    def __getattr__(self, k):
        if self._component_attr_filter(k):
            attr = getattr(self, self._default, None)
            if attr is not None: return getattr(attr,k)
        # raise AttributeError(k)

    def __dir__(self): 
        return custom_dir(self,self._dir())

#     def __getstate__(self): return self.__dict__
    def __setstate__(self,data): 
        self.__dict__.update(data)



def get_device(use_cuda=True, device_id=None, usage=5):
    "Return or set default device; `use_cuda`: None - CUDA if available; True - error if not available; False - CPU"
    if not torch.cuda.is_available():
        use_cuda = False
    else:
        if device_id is None: 
            device_ids = get_available_cuda(usage=usage)
            device_id = device_ids[0]   # get the first available device 
        torch.cuda.set_device(device_id)
    return torch.device(torch.cuda.current_device()) if use_cuda else torch.device('cpu')


def set_device(usage=5):    
    "set the device that has usage < default usage  "
    device_ids = get_available_cuda(usage=usage)
    torch.cuda.set_device(device_ids[0])   # get the first available device


def default_device(use_cuda=True):
    "Return or set default device; `use_cuda`: None - CUDA if available; True - error if not available; False - CPU"
    if not torch.cuda.is_available():
        use_cuda = False
    return torch.device(torch.cuda.current_device()) if use_cuda else torch.device('cpu')


def get_available_cuda(usage=10):
    if not torch.cuda.is_available(): return
    # collect available cuda devices, only collect devices that has less that 'usage' percent 
    device_ids = []
    for device in range(torch.cuda.device_count()):
        if torch.cuda.utilization(device) < usage: device_ids.append(device)
    return device_ids



def to_device(b, device=None, non_blocking=False):
    """
    Recursively put `b` on `device`
    components of b are torch tensors
    """
    if device is None: 
        device = default_device(use_cuda=True)

    if isinstance(b, dict):
        return {key: to_device(val, device) for key, val in b.items()}

    if isinstance(b, (list, tuple)):        
        return type(b)(to_device(o, device) for o in b)      
    
    return b.to(device, non_blocking=non_blocking)


def to_numpy(b):
    """
    Components of b are torch tensors
    """
    if isinstance(b, dict):
        return {key: to_numpy(val) for key, val in b.items()}

    if isinstance(b, (list, tuple)):
        return type(b)(to_numpy(o) for o in b)

    return b.detach().cpu().numpy()

