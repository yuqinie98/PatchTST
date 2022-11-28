from .core import Callback
import torch
from torch.utils.data import DistributedSampler, DataLoader, SequentialSampler
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DistributedTrainer(Callback):
    "Wrap `model` in `DistributedDataParallel` and `dls` in `DistributedDL`"
    def __init__(self,
                 local_rank,
                 world_size,
                 sync_bn=True,  # Whether to replace all batch norm with `nn.SyncBatchNorm`
                 **kwargs
                 ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.sync_bn = sync_bn
        self.kwargs = kwargs
        super().__init__()

    def before_fit(self):
        self.learner.model = self.prepare_model(
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model) if self.sync_bn else self.model,
            ddp_kwargs=self.kwargs
        )
        self.old_train_dl = self.dls.train
        self.old_valid_dl = self.dls.valid

        self.learner.dls.train = self._wrap_dl(self.dls.train)
        self.learner.dls.valid = self._wrap_dl(self.dls.valid)

    def _wrap_dl(self, dl):
        return dl if isinstance(dl, DistributedDL) else self.prepare_data_loader(dl)


    def after_fit(self): 
        self.learner.model = self.learner.model.module 
        self.learner.dls.train = self.old_train_dl
        self.learner.dls.valid = self.old_valid_dl

    def prepare_model(self,
                      model: torch.nn.Module,
                      move_to_device: bool = True,
                      wrap_ddp: bool = True,
                      ddp_kwargs: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """Prepares the model for distributed execution.
        Args:
            model (torch.nn.Module): A torch model to prepare.
            move_to_device (bool): Whether to move the model to the correct
                device. If set to False, the model needs to manually be moved
                to the correct device.
            wrap_ddp (bool): Whether to wrap models in
                ``DistributedDataParallel``.
            ddp_kwargs (Dict[str, Any]): Args to pass into
                ``DistributedDataParallel`` initialization if ``wrap_ddp`` is
                set to True.
        """
        ddp_kwargs = ddp_kwargs or {}

        rank = self.local_rank
        device = torch.device(f"cuda:{rank}")

        # device = get_device()

        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        if move_to_device:
            logger.info(f"Moving model to device: {device}")
            model = model.to(device)
        if wrap_ddp and self.world_size > 1:
            logger.info("Wrapping provided model in DDP.")
            if torch.cuda.is_available():
                model = DistributedDataParallel(
                    model, device_ids=[rank], output_device=rank, **ddp_kwargs)
            else:
                model = DistributedDataParallel(model, **ddp_kwargs)

        return model

    def prepare_data_loader(self,
                            data_loader: torch.utils.data.DataLoader,
                            add_dist_sampler: bool = True,
                            move_to_device: bool = True) -> torch.utils.data.DataLoader:
        """
        Prepares DataLoader for distributed execution.

        This allows you to use the same exact code regardless of number of
        workers or the device type being used (CPU, GPU).

        Args:
            data_loader (torch.utils.data.DataLoader): The DataLoader to
                prepare.
            add_dist_sampler (bool): Whether to add a DistributedSampler to
                the provided DataLoader.
            move_to_device (bool): If set, automatically move the data
                returned by the data loader to the correct device.
        """

        # Only add Distributed Sampler if the following conditions hold:
        # 1. More than one training worker is being used.
        # 2. A DistributedSampler has not already been added by the user.
        # 3. The dataset is not an IterableDataset. Samplers do not worker with
        # IterableDatasets.
        def with_sampler(loader):
            # Automatically set the DistributedSampler

            # If using a sampler, the shuffle attribute in the
            # DataLoader must be set to False.
            # Instead the shuffling is determined by the shuffle attribute
            # in the DistributedSampler.
            # We identify if shuffling is enabled in the passed in
            # DataLoader by seeing if the sampler for the DataLoader is a
            # SequentialSampler.
            shuffle = not isinstance(loader.sampler, SequentialSampler)

            data_loader_args = {
                "dataset": loader.dataset,
                "batch_size": loader.batch_size,
                "shuffle": False,
                "num_workers": loader.num_workers,
                "collate_fn": loader.collate_fn,
                "pin_memory": loader.pin_memory,
                "drop_last": loader.drop_last,
                "timeout": loader.timeout,
                "worker_init_fn": loader.worker_init_fn,
                "sampler": DistributedSampler(loader.dataset, shuffle=shuffle)
            }
            return DataLoader(**data_loader_args)

        data_loader = with_sampler(data_loader)

        if move_to_device:
            if torch.cuda.is_available():
                rank = self.local_rank
                device = torch.device(f"cuda:{rank}")
            else:
                device = torch.device("cpu")
            data_loader = DistributedDL(data_loader, device)

        return data_loader


class DistributedDL(DataLoader):
    def __init__(self, base_dataloader: DataLoader, device: torch.device):

        self.__dict__.update(getattr(base_dataloader, "__dict__", {}))
        self.dataloader = base_dataloader
        self.device = device

    def _move_to_device(self, item):
        def try_move_device(i):
            try:
                i = i.to(self.device)
            except AttributeError:
                logger.debug(f"Item {i} cannot be moved to device "
                             f"{self.device}.")
            return i

        return tuple(try_move_device(i) for i in item)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        iterator = iter(self.dataloader)

        for item in iterator:
            yield self._move_to_device(item)
