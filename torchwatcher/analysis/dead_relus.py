import torch

from torchwatcher.analysis.analysis import Analyzer


class DeadReLU(Analyzer):
    def process_batch_state(self, name, state, working_results):
        x = state.outputs

        if working_results is None:
            working_results = torch.zeros(x.shape[1:])

        with torch.no_grad():
            working_results += (x.detach().sum(dim=0) > 0).float()

        return working_results

    def result_to_dict(self, result) -> dict:
        return {
            'dead_count': result.numel() - result.sum(),
            'numel': result.numel(),
            'sum': result.sum()
        }
