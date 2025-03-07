import torch

from src.torchwatcher.analysis.analysis import Analyzer


class DeadReLU(Analyzer[torch.Tensor]):
    """
    Track the number of dead ReLU (or other saturating) activations over
    batches of data.
    """
    def process_batch_state(self, name, state, working_results):
        x = state.outputs

        if working_results is None:
            working_results = torch.zeros(x.shape[1:])

        with torch.no_grad():
            working_results += (x.detach().sum(dim=0) > 0).float()

        return working_results

    def finalise_result(self, name, result) -> dict:
        return {
            'dead_count': result.numel() - result.sum(),
            'numel': result.numel(),
            'sum': result.sum()
        }
