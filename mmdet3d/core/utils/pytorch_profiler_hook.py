from mmcv.runner import HOOKS, Hook
import torch

@HOOKS.register_module()
class PytorchProfilerHook(Hook):
    def __init__(self, out_dir='./log/profiler'):
        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
            record_shapes=True,
            with_stack=True)
    
    def before_epoch(self, runner):
        # print("start one epoch..")
        self.prof.start()
        return super().before_epoch(runner)
    
    def after_epoch(self, runner):
        # print("finish one epoch..")
        self.prof.stop()
        return super().after_epoch(runner)

    # def before_iter(self, runner):
    #     # print("start one iter..")
    #     return super().before_iter(runner)
    
    def after_iter(self, runner):
        # print("finish one iter..")
        self.prof.step()
        return super().after_iter(runner)