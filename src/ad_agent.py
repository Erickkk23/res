'''
ad_engine.py
Advertisement Selection Engine that employs a Decision Network
to Maximize Expected Utility associated with different Decision
variables in a stochastic reasoning environment.
'''
import math
import itertools
import unittest
import numpy as np
import pandas as pd
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

class AdEngine:

    def __init__(self, data: "pd.DataFrame", structure: list[tuple[str, str]], dec_vars: list[str], util_map: dict[str, dict[int, int]]):

        self._data = data
        self._structure = structure
        self._dec_vars = dec_vars
        self._util_map = util_map

        self._model = BayesianNetwork(structure)
        self._model.fit(data)

        self._causal_infer = CausalInference(self._model)
        self._elimination = VariableElimination(self._model)

        self._var_vals = {
            col: sorted(data[col].unique().tolist())
            for col in data.columns
        }

        self._util_vars = list(util_map.keys())

        all_vars = set(data.columns)
        self._chance_vars = list(all_vars - set(dec_vars) - set(self._util_vars))
        
    def meu(self, evidence: dict[str, int]) -> tuple[dict[str, int], float]:
        best_assignment = {}
        best_utility = float("-inf")

        for decision_assignment in self._generate_decision_combos():
            utility = self._expected_utility(decision_assignment, evidence)

            if utility > best_utility:
                best_utility = utility
                best_assignment = decision_assignment

        return best_assignment, best_utility


    def _generate_decision_combos(self) -> list[dict[str, int]]:
        combos = itertools.product(*[self._var_vals[var] for var in self._dec_vars])
        return [dict(zip(self._dec_vars, combo)) for combo in combos]


    def _expected_utility(self, decision: dict[str, int], evidence: dict[str, int]) -> float:
        total_utility = 0.0

        for util_var, util_values in self._util_map.items():
            query = self._causal_infer.query(
                [util_var],
                do=decision,
                evidence=evidence,
                show_progress=False
            )

            for val_index, prob in enumerate(query.values):
                total_utility += prob * util_values.get(val_index, 0)

        return total_utility


    def vpi(self, potential_evidence: str, observed_evidence: dict[str, int]) -> float:
        """
        Computes the Value of Perfect Information (VPI) for a potential variable
        given currently observed evidence.

        Args:
            potential_evidence (str): The variable we are considering learning.
            observed_evidence (dict[str, int]): The current known traits of the user.

        Returns:
            float: Expected increase in utility from knowing the variable.
        """
        _, base_meu = self.meu(observed_evidence)
        expected_meu_with_info = 0.0

        for val in self._var_vals[potential_evidence]:
            extended_evidence = observed_evidence.copy()
            extended_evidence[potential_evidence] = val

            _, meu_with_val = self.meu(extended_evidence)

            dist = self._elimination.query(
                [potential_evidence],
                evidence=observed_evidence,
                show_progress=False
            )
            prob = dist.values[val]
            expected_meu_with_info += prob * meu_with_val

        return expected_meu_with_info - base_meu

    
    def most_likely_consumer(self, evidence: dict[str, int]) -> dict[str, int]:
        """
        Fills in the most likely values for all unknown variables *excluding* decision vars.
        Observed evidence values are preserved exactly.
        """
        result = self._elimination.map_query(
            variables=None,
            evidence=evidence,
            show_progress=False
        )

        completed = dict(evidence)

        for var, val in result.items():
            if var not in evidence and var not in self._dec_vars:
                completed[var] = int(val)

        return completed

