import torch
import t2v_metrics
from time import time
from dataclasses import dataclass
from typing import Literal

@dataclass
class ValidationResult:
    method: str
    model_name: str
    score: float
    model_size_mb: float
    inference_time_sec: float


class TextImageValidator:
    def __init__(self, method: Literal['clip', 'blip', 'vqa'], model_name: str = '', debug: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug = debug
        self.method = method
        self.model_name = model_name
        self.validator = self.load_model(model_name)
        self.model_size = self.get_model_size()

    def load_model(self, model_name):
        if self.method == 'clip':
            self.validator = t2v_metrics.CLIPScore(model=model_name, device=self.device)
        elif self.method == 'blip':
            self.validator = t2v_metrics.ITMScore(model=model_name, device=self.device)
        else:
            self.validator = t2v_metrics.VQAScore(model=model_name, device=self.device)
        return self.validator

    def evaluate(self, image_path, text):
        t1 = time()
        with torch.no_grad(), torch.amp.autocast(self.device):
            score = self.validator(images=image_path, texts=text).item()
        t2 = time()
        if self.debug:
            print(f'{self.method} score: {score}')
            print(f'Inference time: {t2 - t1} sec')
        return ValidationResult(
            method=self.method,
            model_name=self.model_name,
            score=score,
            model_size_mb=self.model_size,
            inference_time_sec=t2 - t1
        )

    def del_model(self):
        del self.validator
        torch.cuda.empty_cache()

    def get_model_size(self):
        model = self.validator.model.model
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        model_size_mb = (param_size + buffer_size) / 1024 ** 2
        if self.debug:
            print('model size: {:.3f} MB'.format(model_size_mb))
        return model_size_mb

    def get_list_of_pretrained_models(self):
        if self.method == 'clip':
            return t2v_metrics.list_all_clipscore_models()
        elif self.method == 'blip':
            return t2v_metrics.list_all_itmscore_models()
        else:
            return t2v_metrics.list_all_vqascore_models()
