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
        """
        Responsible for initializing the Decision Network of the
        AdEngine by taking in the dataset, structure of network,
        any decision variables, and a map of utilities
        
        Parameters:
            data (pd.DataFrame):
                Pandas data frame containing all data on which the decision
                network's chance-node parameters are to be learned
            structure (list[tuple[str, str]]):
                The Bayesian Network's structure, a list of tuples denoting
                the edge directions where each tuple is (parent, child)
            dec_vars (list[str]):
                list of string names of variables to be
                considered decision variables for the agent. Example:
                ["Ad1", "Ad2"]
            util_map (dict[str, dict[int, int]]):
                Discrete, tabular, utility map whose keys
                are variables in network that are parents of a utility node, and
                values are dictionaries mapping that variable's values to a utility
                score, for example:
                  {
                    "X": {0: 20, 1: -10}
                  }
                represents a utility node with single parent X whose value of 0
                has a utility score of 20, and value 1 has a utility score of -10
        """
        return
        
    def meu(self, evidence: dict[str, int]) -> tuple[dict[str, int], float]:
        """
        Computes the Maximum Expected Utility (MEU) defined as the choice of
        decision variable values that maximize expected utility of any evaluated
        chance nodes given in the agent's utility map.
        
        Parameters:
            evidence (dict[str, int]):
                dict mapping network variables to their observed values, 
                of the format: {"Obs1": val1, "Obs2": val2, ...}
        
        Returns: 
            tuple[dict[str, int], float]:
                A 2-tuple of the format (a*, MEU) where:
                [0] is a dictionary mapping decision variables to their MEU states
                [1] is the MEU value (a float) of that decision combo
        """
        # [!] TODO
        return ({"TODO": 0}, -1.0)

    def vpi(self, potential_evidence: str, observed_evidence: dict[str, int]) -> float:
        """
        Given some observed demographic "evidence" about a potential
        consumer, returns the Value of Perfect Information (VPI)
        that would be received on the given "potential" evidence about
        that consumer.
        
        Parameters:
            potential_evidence (str):
                string representing the variable name of the variable 
                under consideration for potentially being obtained
            observed_evidence (tuple[dict[str, int], float]):
                dict mapping network variables 
                to their observed values, of the format: 
                {"Obs1": val1, "Obs2": val2, ...}
        
        Returns:
            float:
                float value indicating the VPI(potential | observed)
        """
        # [!] TODO
        return -1
    
    def most_likely_consumer(self, evidence: dict[str, int]) -> dict[str, int]:
        """
        Given some known traits about a particular consumer, makes the best guess
        of the values of any remaining hidden variables and returns the completed
        data point as a dictionary of variables mapped to their most likely values.
        (Observed evidence will always have the same values in the output).
        
        Parameters:
            evidence (dict[str, int]):
                dict mapping network variables 
                to their observed values, of the format: 
                {"Obs1": val1, "Obs2": val2, ...}
        
        Returns:
            dict[str, int]:
                The most likely values of all variables given what's already
                known about the consumer.
        """
        # [!] TODO!
        return {"TODO": -1}
    