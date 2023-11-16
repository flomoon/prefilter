from pathlib import Path
from absl import logging

import torch


class Checkpoint:
    def __init__(self, dir):
        self.base_path = Path(dir)
        self._get_step_fn = lambda x: int(x.name.split('_')[-1].split('.')[0])
        

    def save(self, net, optim=None, step=0, persistent_step_period=None):
        self.base_path.mkdir(parents=True, exist_ok=True)
        ckpt_file = self.base_path / f'checkpoint_{step}.pth'

        saving_obj = dict(step=step, model=net.state_dict())
        if optim:
            saving_obj.update(dict(optimizer=optim.state_dict()))

        if persistent_step_period:
            old_ckpts = self.base_path.glob('*.pth')
            for ckpt in old_ckpts:
                if int(ckpt.name.split('_')[-1][:-4]) % persistent_step_period:
                    ckpt.unlink()

        torch.save(saving_obj, ckpt_file)
        logging.info(f"Weight is saved ({step} steps).")

    def load(self, net, optim=None, step=-1):
        self._ckpt_files = sorted(self.base_path.glob('*.pth'), key=self._get_step_fn)
        if len(self._ckpt_files):
            stored_steps = list(map(self._get_step_fn, self._ckpt_files))
            logging.info(f"There is(are) {len(self._ckpt_files)} checkpoint(s), step(s): {stored_steps}")
            if step == -1:
                target_ckpt_file = self._ckpt_files[step]
            else:
                target_ckpt_file = self.base_path / f'checkpoint_{step}.pth'
            current_ckpt = torch.load(target_ckpt_file, map_location=torch.device('cpu'))

            net.load_state_dict(current_ckpt['model'])
            if optim:
                optim.load_state_dict(current_ckpt['optimizer'])

            step = current_ckpt['step']
            logging.info(f"Weight is loaded from step: {step}.")
        else:
            step = 0
        return step

    def resume(self, net, optim, idx=-1): 
        step = self.load(net, optim, idx)
        if step == 0:
            logging.info("Start training from the scratch.")
        else:
            logging.info(f"Resume training from step: {step+1}.")
        return step