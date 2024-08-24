from training import ProgressCounter
from training.AbstractEvaluatorTrainer import AbstractTrainer, AbstractEvaluator


def train_model(trainer: AbstractTrainer, dev_evaluator: AbstractEvaluator, test_evaluator: AbstractEvaluator, epochs, save_metric_min_keys=[], save_metric_max_keys=[], objs_to_save_after_ep={}):
    """
    :param trainer: Trainer object
    :param evaluator: Evaluator object
    :param epochs: number of epochs to train
    :param save_metric_min_keys: save model with min metric (saved using evaluator.save_model)
    :param save_metric_max_keys: save model with max metric (saved using evaluator.save_model)
    :param objs_to_save_after_ep: objects are saved using trainer log_manager, key and epoch identifier is used for filename
    """
    metric_comparator = MetricComparator(save_metric_min_keys, save_metric_max_keys)
    maxmin = StoreMaxMin()

    maxmineval = StoreMaxMin()
    # evalmetric_comparator = MetricComparator([], ["semeval_weighted_f1"])

    for _ep in range(epochs):
        train_res = trainer.epoch()     # also logs results and saves the checkpoint
        val_res = dev_evaluator.epoch()

        maxmin.update(val_res)
        dev_evaluator._log_manager.log_dev(maxmin.get_mins())
        dev_evaluator._log_manager.log_dev(maxmin.get_maxs())

        # test_res = evaluate(trainer._model, trainer._dataloader.dataset, res_json=trainer._log_manager.dir + f"/model_ep_{ProgressCounter.TRAIN_EPOCH}.json")
        # trainer._log_manager.log_test(test_res)
        # maxmineval.update(test_res)
        # trainer._log_manager.log_test(maxmineval.get_mins())
        # trainer._log_manager.log_test(maxmineval.get_maxs())

        # if validation results are better than previous ones
        if metric_comparator.is_better(val_res):
            dev_evaluator.save_model(f"model_ep_{ProgressCounter.TRAIN_EPOCH}")

        for k, obj in objs_to_save_after_ep.items():
            trainer._log_manager.checkpoint(f"{k}_ep_{ProgressCounter.TRAIN_EPOCH}", obj)

    # final testing
    if test_evaluator is not None:
        print("FINAL TEST with Test data")
        test_res = test_evaluator.epoch()
        test_evaluator._log_manager.log_test(test_res)

class MetricComparator:
    def __init__(self, save_metric_min_keys, save_metric_max_keys):
        self.mins = {}
        self.maxs = {}
        self.minkeys = save_metric_min_keys
        self.maxkeys = save_metric_max_keys
        for k in self.minkeys:
            self.mins[k] = float("inf")
        for k in self.maxkeys:
            self.maxs[k] = float("-inf")

    def is_better(self, res):
        rtn = False
        for k in self.minkeys:
            if res[k] < self.mins[k]:
                self.mins[k] = res[k]
                rtn = True
        for k in self.maxkeys:
            if res[k] > self.maxs[k]:
                self.maxs[k] = res[k]
                rtn = True
        return rtn


class StoreMaxMin:
    def __init__(self):
        self.mins = None
        self.maxs = None

    def update(self, res: dict):
        if self.mins is None:
            self.mins = res.copy()
            self.maxs = res.copy()
        else:
            for k, v in res.items():
                self.mins[k] = min(self.mins[k], res[k])
                self.maxs[k] = max(self.maxs[k], res[k])

    def get_mins(self):
        return {f"min_{k}": v for k, v in self.mins.items()}

    def get_maxs(self):
        return {f"max_{k}": v for k, v in self.maxs.items()}