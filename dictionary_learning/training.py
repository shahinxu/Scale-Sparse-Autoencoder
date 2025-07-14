import torch as t
from .dictionary import AutoEncoder
import os
from tqdm import tqdm
from .trainers.standard import StandardTrainer
import json
from .utils import cfg_filename
import gc
import logging
from datetime import datetime

def trainSAE(
        data, 
        trainer_configs = [
            {
                'trainer' : StandardTrainer,
                'dict_class' : AutoEncoder,
                'activation_dim' : 512,
                'dictionary_size' : 64*512,
                'lr' : 1e-3,
                'l1_penalty' : 1e-1,
                'warmup_steps' : 1000,
                'resample_steps' : None,
                'seed' : None,
                'wandb_name' : 'StandardTrainer',
            }
        ],
        log_steps=None,
        steps=None,
        save_dir=None,
        activations_split_by_head=False,
):
    trainers = []
    for config in trainer_configs:
        trainer = config['trainer']
        del config['trainer']
        trainers.append(trainer(**config))

    if save_dir is not None:
        save_dirs = [os.path.join(save_dir, f"{cfg_filename(trainer_config)}") for trainer_config in trainer_configs]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            config = {'trainer' : trainer.config}
            try:
                config['buffer'] = data.config
            except: pass
            with open(os.path.join(dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]
    
    log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            # logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    for step, act in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            break
        if log_steps is not None and step % log_steps == 0:
            with t.no_grad():
                z = act.clone()
                for i, trainer in enumerate(trainers):
                    act = z.clone()
                    if activations_split_by_head:
                        act = act[..., i, :]
                    trainer_name = f'{trainer.config["wandb_name"]}-{i}'
                    
                    act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                    logger.info(f"Step {step} | {trainer_name} | Losslog: {losslog}")
                        
        for trainer in trainers:
            trainer.update(step, act)
        del act, act_hat, f, losslog
        gc.collect()
        t.cuda.empty_cache()

    for save_dir, trainer in zip(save_dirs, trainers):
        if save_dir is not None:
            t.save(trainer.ae.state_dict(), os.path.join(save_dir, "ae.pt"))