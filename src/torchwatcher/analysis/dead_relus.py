import torch

from torchwatcher.analysis.analysis import Analyser


class DeadReLU(Analyser[torch.Tensor]):
    """
    Track the number of dead ReLU (or other saturating) activations over
    batches of data.
    """
    def process_batch_state(self, name, state, working_results):
        with torch.no_grad():
            active = state.outputs.detach().gt(0).any(dim=0)
            if working_results is None:
                working_results = active
            else:
                working_results.logical_or_(active)

        return working_results

    def finalise_result(self, name, result) -> dict:
        active_count = result.count_nonzero()
        return {
            'dead_count': result.numel() - active_count,
            'numel': result.numel(),
            'sum': active_count
        }
