import json
import os

import torch
import wandb

from training import ProgressCounter
import config
from utils import unique_path


class AbstractManager:
    def __init__(self, name, new_path=True):
        pth = os.path.join(config.paths.train_bin, name)
        if new_path:
            pth = unique_path(pth)

        self.name = os.path.split(pth)[1]
        self.dir = pth
        os.makedirs(self.dir, exist_ok=not new_path)

    def checkpoint(self, fname, model, optimizer=None):
        torch.save(model.state_dict(), os.path.join(self.dir, fname + ".sd"))
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(self.dir, fname + "_opt.sd"))

    def log_step(self, res):
        raise NotImplementedError

    def log_train(self, res):
        raise NotImplementedError

    def log_dev(self, res):
        raise NotImplementedError

    def log_test(self, res):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class VoidManager(AbstractManager):
    def log_step(self, res):
        pass

    def log_train(self, res):
        pass

    def log_dev(self, res):
        pass

    def close(self):
        pass


class JsonManager(AbstractManager):
    def __init__(self, name, run_config, step_log=True, new_path=True):
        super().__init__(name, new_path)
        self.step_log = step_log
        self.train_path = os.path.join(self.dir, "train.json")
        self.dev_path = os.path.join(self.dir, "dev.json")
        self.test_path = os.path.join(self.dir, "test.json")
        self.step_path = os.path.join(self.dir, "step.json")

        with open(os.path.join(self.dir, "run_config.json"), "w") as f:
            json.dump(run_config, f, indent=4)

        with open(self.train_path, "w") as f:
            f.write("[")
        with open(self.dev_path, "w") as f:
            f.write("[")
        if self.step_log:
            with open(self.step_path, "w") as f:
                f.write("[")

    def _log_file(self, data, fpath):
        with open(fpath, "a") as f:
            s = json.dumps(data, indent=8)
            f.write(s + ", ")

    def log_step(self, res):
        if self.step_log:
            self._log_file({"res": res, "step": ProgressCounter.TRAIN_STEP}, self.step_path)

    def log_dev(self, res):
        self._log_file({"res_dev": res, "epoch": ProgressCounter.TRAIN_EPOCH, "step": ProgressCounter.TRAIN_STEP}, self.dev_path)

    def log_test(self, res):
        self._log_file({"res_test": res, "epoch": ProgressCounter.TRAIN_EPOCH, "step": ProgressCounter.TRAIN_STEP}, self.test_path)

    def log_train(self, res):
        self._log_file({"res_train": res, "epoch": ProgressCounter.TRAIN_EPOCH, "step": ProgressCounter.TRAIN_STEP}, self.train_path)

    def close(self):
        with open(self.train_path, "a") as f:
            f.write("]")
        with open(self.dev_path, "a") as f:
            f.write("]")
        if self.step_log:
            with open(self.step_path, "a") as f:
                f.write("]")


class WandBManager(AbstractManager):
    def __init__(self, name, run_config, new_path=True):
        super().__init__(name, new_path)
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=run_config,
            name=self.name,
            dir=self.dir
        )
        # Define the custom x axis metric
        wandb.define_metric("TRAIN_EPOCH")
        wandb.define_metric("TRAIN_STEP")

        # Define which metrics to plot against that x-axis
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="TRAIN_EPOCH")
        # set all other dev/ metrics to use this step
        wandb.define_metric("dev/*", step_metric="TRAIN_EPOCH")
        # set all other dev/ metrics to use this step
        wandb.define_metric("step/*", step_metric="TRAIN_STEP")

        wandb.define_metric("test/*", step_metric="TRAIN_EPOCH")

    def _prepend_key(self, res, prepkey):
        d = {}
        for k, v in res.items():
            d[f"{prepkey}/{k}"] = v
        return d

    def log_step(self, res):
        data = self._prepend_key(res, "step")
        data["TRAIN_STEP"] = ProgressCounter.TRAIN_STEP
        wandb.log(data)

    def log_dev(self, res):
        data = self._prepend_key(res, "dev")
        data["TRAIN_EPOCH"] = ProgressCounter.TRAIN_EPOCH
        data["TRAIN_STEP"] = ProgressCounter.TRAIN_STEP
        wandb.log(data)

    def log_test(self, res):
        data = self._prepend_key(res, "test")
        data["TRAIN_EPOCH"] = ProgressCounter.TRAIN_EPOCH
        data["TRAIN_STEP"] = ProgressCounter.TRAIN_STEP
        wandb.log(data)

    def log_train(self, res):
        data = self._prepend_key(res, "train")
        data["TRAIN_EPOCH"] = ProgressCounter.TRAIN_EPOCH
        data["TRAIN_STEP"] = ProgressCounter.TRAIN_STEP
        wandb.log(data)

    def close(self):
        wandb.finish()


class JsonWandBManager(AbstractManager):
    def __init__(self, dir, run_config, new_path=True):
        super().__init__(dir, new_path)
        self.jsonmanager = JsonManager(self.name, run_config, step_log=False, new_path=False)
        self.wandbmanager = WandBManager(self.name, run_config, new_path=False)

    def log_step(self, res):
        self.jsonmanager.log_step(res)
        self.wandbmanager.log_step(res)

    def log_train(self, res):
        self.jsonmanager.log_train(res)
        self.wandbmanager.log_train(res)

    def log_dev(self, res):
        self.jsonmanager.log_dev(res)
        self.wandbmanager.log_dev(res)

    def log_test(self, res):
        self.jsonmanager.log_test(res)
        self.wandbmanager.log_test(res)

    def close(self):
        self.jsonmanager.close()
        self.wandbmanager.close()


class JsonPredictManager(AbstractManager):
    def __init__(self, name, step_log=True, new_path=True, output_predict_json="pred_test.json"):
        super().__init__(name, new_path)
        self.step_log = step_log
        self.test_path = os.path.join(output_predict_json)

    def _log_file(self, data, fpath):
        with open(fpath, "a") as f:
            s = json.dumps(data, indent=8)
            f.write(s + ", ")

    def log_test(self, res):
        self._log_file({"res_test": res, "epoch": ProgressCounter.TRAIN_EPOCH, "step": ProgressCounter.TRAIN_STEP}, self.test_path)