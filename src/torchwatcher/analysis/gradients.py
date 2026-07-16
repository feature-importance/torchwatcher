from torchwatcher.analysis.analysis import Analyser, AnalyserState
from torchwatcher.analysis.running_stats import Variance

class GradientAccumulationAnalyser(Analyser):
    def __init__(self):
        super().__init__(gradient=True)

    def process_batch_state(self,
                        name: str,
                        state: AnalyserState,
                        working_results: Variance | None) -> Variance:
        if working_results is None:
            working_results = Variance()

        gradients = state.output_gradients.mean(dim=(-2,-1))
        working_results.add(gradients)

        return working_results

    def finalise_result(self, name: str, result: Variance) -> dict:
        rec = dict()
        rec['var'] = result.variance()
        rec['mean'] = result.mean()
        return rec