import numpy as np
from tqdm import tqdm

from training import ProgressCounter
from training.Manager import AbstractManager


class AbstractEvaluator:
    """inherit and implement _step function"""
    def __init__(self, model, dataloader, log_manager: AbstractManager):
        """
        :param model: pytorch model
        :param dataloader: iterable, the item will be provided in step function
        :param log_manager: to log progress and save model
        """
        self._model = model
        self._dataloader = dataloader
        self._log_manager = log_manager

    def _step(self, data):
        """
        :param data: item from dataloader
        :return: dict with metrics and losses per step
        """
        raise NotImplementedError
        # return {"metric": 0.5}

    def _step_after(self, step_res):
        # no step logging in evaluator
        pass

    def step(self, data):
        res = self._step(data)
        self._step_after(res)
        return res

    def epoch(self):
        self._epoch_before()
        res = self._epoch()
        self._epoch_after(res)
        return res

    def _epoch_before(self):
        self._model.train(False)

    def _epoch(self):
        res = []
        pbar = tqdm(self._dataloader, desc=f"{self.__class__.__name__} epoch {ProgressCounter.TRAIN_EPOCH}")
        for data in pbar:
            stepres = self.step(data)
            pbar.set_postfix(stepres)
            res.append(stepres)

        aggres = self.aggregate_res(res)
        tmp = [f"{k}={v}" for k, v in aggres.items()]
        print(", ".join(tmp))
        pbar.set_postfix(aggres)
        pbar.close()
        return aggres

    def aggregate_res(self, res):
        aggres = {}
        for k in list(res[0].keys()):
            aggres[k] = np.mean([r[k] for r in res])
        return aggres

    def _epoch_after(self, res):
        self._log_manager.log_dev(res)

    def save_model(self, fname):
        self._log_manager.checkpoint(fname, self._model)


class AbstractTrainer(AbstractEvaluator):
    """inherit and implement _step function"""
    def __init__(self, model, dataloader, optimizer, log_manager, log_each_step=1000):
        super().__init__(model, dataloader, log_manager)
        optimizer.zero_grad()
        self._optimizer = optimizer
        self._log_each_step = log_each_step

    def _step(self, data):
        """
        :param data: item from dataloader
        :return: dict with metrics and losses per step
        """
        raise NotImplementedError
        # return {"metric": 0.5}

    def _step_after(self, step_res):
        super()._step_after(step_res)
        if ProgressCounter.TRAIN_STEP % self._log_each_step == 0:
            self._log_manager.log_step(step_res)

    def _epoch_before(self):
        self._model.train(True)

    def _epoch_after(self, res):
        self.checkpoint("last_model")
        self._log_manager.log_train(res)

    def checkpoint(self, fname):
        self._log_manager.checkpoint(fname, self._model, self._optimizer)

    def step(self, data):
        ProgressCounter.TRAIN_STEP += 1
        return super().step(data)

    def epoch(self):
        ProgressCounter.TRAIN_EPOCH += 1
        return super().epoch()