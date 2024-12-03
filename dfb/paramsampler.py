from scipy import stats

from typing import List, Dict

class LogScaleSampler:

    def __init__(self,
                 n_params: int,
                 param_names: List[str],
                 lb: List[float],
                 ub: List[float],
                 reversed: List[bool],
                 n_exps: int) -> None:
        if len(lb) != n_params or len(ub) != n_params or len(param_names) != n_params:
            raise ValueError("Parameter dimension is not matched")
        if n_exps < 0:
            raise ValueError("Number of experiments must be positive")

        self._sampler = stats.qmc.Halton(d=n_params, scramble=False)
        self._reversed = reversed
        self._n_params = n_params
        self._n_exps = n_exps
        self._lb = lb
        self._ub = ub
        self._param_names = param_names
    
    def create_params(self) -> Dict:
        _ = self._sampler.fast_forward(self._n_exps)
        samples = self._sampler.random(self._n_exps)
        q_samples = stats.qmc.scale(samples, self._lb, self._ub)

        param_map = {}

        for i in range(self._n_params):
            p = 10 ** q_samples[:, i]
            if self._reversed[i]:
                p = 1 - p
            param_map[self._param_names[i]]  = p

        return param_map

