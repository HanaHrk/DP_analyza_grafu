from training import ProgressCounter
from training.AbstractEvaluatorTrainer import AbstractTrainer, AbstractEvaluator
from sklearn import metrics
import os
import torch


def evalstep(model, data, loss_function, loss_weight=1, compute_metrics=True):

    classes_gt = data[-1].to(model.device)
    # utt_classes_gt = data[-2].to(model.device)
    # utt_links_gt = data[-1].to(model.device)
    # utt_classes_logits, utt_links_logits = model(*data)
    # data[0] = input_ids, data[1] = attention masks

    # necceasary --> model expects float32 instead of float64
    classes_logits = model(data[0].float().to(model.device))

    closs = loss_function(
        classes_logits.reshape((-1, classes_logits.shape[-1])),
        classes_gt.flatten()
    )
    # lloss = LINK_LOSS(
    #     utt_links_logits,
    #     utt_links_gt
    # )

    # loss = CLASSW * closs + LINKW * lloss
    loss = loss_weight * closs

    mdict = {
        "loss": loss.detach().cpu().item(),
        # "class_loss": closs.detach().cpu().item(),
        # "link_loss": lloss.detach().cpu().item(),
    }
    if compute_metrics:
        cgt = classes_gt.detach().cpu().flatten()
        cpred = classes_logits.argmax(dim=-1).detach().cpu().flatten()
        # lgt = (utt_links_gt > 0.5).detach().cpu().flatten()
        # lpred = (utt_links_logits > 0.0).detach().cpu().flatten()   # logits zero is 0.5 after sigmoid, can not be 0 due to masking

        mdict["acc"] = metrics.accuracy_score(
            y_true=cgt,
            y_pred=cpred
        )

        mdict["macro_f1"] = metrics.f1_score(
            y_true=cgt,
            y_pred=cpred,
            average="macro"
        )

        mdict["weighted_f1"] = metrics.f1_score(
            y_true=cgt,
            y_pred=cpred,
            average="weighted"
        )

    return loss, mdict


class CustomTrainer(AbstractTrainer):
    """inherit and implement _step function"""
    def __init__(self, model, dataloader, optimizer, loss_function, log_manager, log_each_step=1000, gradient_accumulation_steps=1):
        super().__init__(model, dataloader, optimizer, log_manager, log_each_step)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.loss_function = loss_function
        self.loss_weight = 1
    def _step(self, data):
        loss, m = evalstep(self._model, data, loss_function=self.loss_function, loss_weight=self.loss_weight, compute_metrics=True)

        if self.gradient_accumulation_steps > 1:
            self.gradient_accumulation_step(loss)
        else:
            # Backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        return m

    def gradient_accumulation_step(self, loss):
        # Normalize the Gradients
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        if ProgressCounter.TRAIN_STEP % self.gradient_accumulation_steps == 0:
            self._optimizer.step()
            self._optimizer.zero_grad()


class CustomEvaluator(AbstractEvaluator):
    """inherit and implement _step function"""
    def __init__(self, model, dataloader, loss_function, log_manager):
        super().__init__(model, dataloader, log_manager)
        self.loss_function = loss_function
        self.loss_weight = 1

    def _step(self, data):
        """
        :param data: item from dataloader
        :return: dict with metrics and losses per step
        """
        loss, m = evalstep(self._model, data, loss_function=self.loss_function, loss_weight=self.loss_weight, compute_metrics=True)
        return m

    def save_model(self, fname):
        pth = os.path.join(self._log_manager.dir, fname + ".sd")
        self._model.save_learnable_params(pth)