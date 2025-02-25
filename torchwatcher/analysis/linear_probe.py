from typing import Any

from .analysis import Analyzer, AnalyzerState


class LinearProbe(Analyzer):
    def process_batch_state(self,
                            name: str,
                            state: AnalyzerState,
                            working_results: Any | None):

        if self.training:
        #     forward on the relevant linear layer and return the logits
        else:
        #     compute acc, predictions, etc
        pass

    def finalise_result(self, result) -> dict:
        pass