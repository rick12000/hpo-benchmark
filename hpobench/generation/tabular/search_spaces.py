"""
Search space generation for synthetic benchmarks.

This module provides functionality to generate random search spaces with varying
numbers and types of hyperparameters for synthetic benchmark creation.
"""

import random
import numpy as np
from typing import Dict, Union
import logging

from hpobench.config.types import IntRange, FloatRange, CategoricalRange
from hpobench.config.constants import SyntheticGenerationParameters

logger = logging.getLogger(__name__)


class SearchSpaceGenerator:
    """Generator for random hyperparameter search spaces.
    
    Creates search spaces with varying numbers and types of hyperparameters,
    suitable for synthetic benchmark generation.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the search space generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
    
    def generate_search_space(
        self,
        n_hyperparameters: int = None,
        min_hyperparameters: int = None,
        max_hyperparameters: int = None,
    ) -> Dict[str, Union[IntRange, FloatRange, CategoricalRange]]:
        """Generate a random search space.
        
        Args:
            n_hyperparameters: Number of hyperparameters (if None, randomly chosen)
            min_hyperparameters: Minimum number of hyperparameters
            max_hyperparameters: Maximum number of hyperparameters
            
        Returns:
            Dictionary mapping hyperparameter names to their range specifications
        """
        synthetic_generation = SyntheticGenerationParameters()
        
        if min_hyperparameters is None:
            min_hyperparameters = synthetic_generation.min_hyperparameters
        if max_hyperparameters is None:
            max_hyperparameters = synthetic_generation.max_hyperparameters
            
        if n_hyperparameters is None:
            n_hyperparameters = random.randint(min_hyperparameters, max_hyperparameters)
        
        search_space = {}
        
        for i in range(n_hyperparameters):
            hp_name = f"hp_{i}"
            hp_type = random.choice(['int', 'float', 'categorical'])
            
            if hp_type == 'int':
                search_space[hp_name] = self._generate_int_range()
            elif hp_type == 'float':
                search_space[hp_name] = self._generate_float_range()
            else:  # categorical
                search_space[hp_name] = self._generate_categorical_range()
        
        logger.debug(f"Generated search space with {n_hyperparameters} hyperparameters")
        return search_space
    
    def _generate_int_range(self) -> IntRange:
        """Generate a random integer range.
        
        Returns:
            IntRange with random bounds and log scale setting
        """
        # Choose scale type
        use_log = random.random() < 0.3  # 30% chance of log scale
        
        if use_log:
            # For log scale, use positive ranges
            log_lower = random.uniform(0, 3)  # 10^0 to 10^3
            log_upper = random.uniform(log_lower + 1, log_lower + 4)
            lower = int(10 ** log_lower)
            upper = int(10 ** log_upper)
        else:
            # For linear scale, use various ranges
            range_type = random.choice(['small', 'medium', 'large'])
            if range_type == 'small':
                lower = random.randint(0, 10)
                upper = random.randint(lower + 2, lower + 20)
            elif range_type == 'medium':
                lower = random.randint(0, 100)
                upper = random.randint(lower + 10, lower + 200)
            else:  # large
                lower = random.randint(0, 1000)
                upper = random.randint(lower + 100, lower + 5000)
        
        return IntRange(lower=lower, upper=upper, log=use_log)
    
    def _generate_float_range(self) -> FloatRange:
        """Generate a random float range.
        
        Returns:
            FloatRange with random bounds and log scale setting
        """
        # Choose scale type
        use_log = random.random() < 0.4  # 40% chance of log scale
        
        if use_log:
            # For log scale, use positive ranges
            log_lower = random.uniform(-3, 2)  # 10^-3 to 10^2
            log_upper = random.uniform(log_lower + 1, log_lower + 4)
            lower = 10 ** log_lower
            upper = 10 ** log_upper
        else:
            # For linear scale, use various ranges
            range_type = random.choice(['small', 'medium', 'large', 'unit'])
            if range_type == 'small':
                lower = random.uniform(-10, 10)
                upper = random.uniform(lower + 0.1, lower + 20)
            elif range_type == 'medium':
                lower = random.uniform(-100, 100)
                upper = random.uniform(lower + 1, lower + 200)
            elif range_type == 'large':
                lower = random.uniform(-1000, 1000)
                upper = random.uniform(lower + 10, lower + 5000)
            else:  # unit interval
                lower = 0.0
                upper = 1.0
        
        return FloatRange(lower=float(lower), upper=float(upper), log=use_log)
    
    def _generate_categorical_range(self) -> CategoricalRange:
        """Generate a random categorical range.
        
        Returns:
            CategoricalRange with random choices
        """
        # Decide on number of categories
        n_categories = random.randint(2, 10)
        
        # Decide on category type
        category_type = random.choice(['string', 'integer', 'boolean', 'mixed'])
        
        if category_type == 'string':
            # Generate string categories
            choices = [f"cat_{i}" for i in range(n_categories)]
        elif category_type == 'integer':
            # Generate integer categories
            start = random.randint(0, 10)
            choices = list(range(start, start + n_categories))
        elif category_type == 'boolean':
            # Binary choice
            choices = [True, False]
        else:  # mixed
            # Mix of strings and integers
            choices = []
            for i in range(n_categories):
                if random.random() < 0.5:
                    choices.append(f"opt_{i}")
                else:
                    choices.append(i)
        
        return CategoricalRange(choices=choices)
    
    def search_space_to_dict(
        self, 
        search_space: Dict[str, Union[IntRange, FloatRange, CategoricalRange]]
    ) -> Dict:
        """Convert search space to JSON-serializable dictionary.
        
        Args:
            search_space: Search space with Range objects
            
        Returns:
            Dictionary with serializable representation
        """
        result = {}
        for hp_name, hp_range in search_space.items():
            if isinstance(hp_range, IntRange):
                result[hp_name] = {
                    'type': 'IntRange',
                    'lower': hp_range.lower,
                    'upper': hp_range.upper,
                    'log': hp_range.log,
                }
            elif isinstance(hp_range, FloatRange):
                result[hp_name] = {
                    'type': 'FloatRange',
                    'lower': hp_range.lower,
                    'upper': hp_range.upper,
                    'log': hp_range.log,
                }
            elif isinstance(hp_range, CategoricalRange):
                result[hp_name] = {
                    'type': 'CategoricalRange',
                    'choices': hp_range.choices,
                }
        return result
    
    def dict_to_search_space(
        self, 
        search_space_dict: Dict
    ) -> Dict[str, Union[IntRange, FloatRange, CategoricalRange]]:
        """Convert dictionary back to search space with Range objects.
        
        Args:
            search_space_dict: Dictionary representation of search space
            
        Returns:
            Search space with Range objects
        """
        result = {}
        for hp_name, hp_spec in search_space_dict.items():
            hp_type = hp_spec['type']
            if hp_type == 'IntRange':
                result[hp_name] = IntRange(
                    lower=hp_spec['lower'],
                    upper=hp_spec['upper'],
                    log=hp_spec.get('log', False),
                )
            elif hp_type == 'FloatRange':
                result[hp_name] = FloatRange(
                    lower=hp_spec['lower'],
                    upper=hp_spec['upper'],
                    log=hp_spec.get('log', False),
                )
            elif hp_type == 'CategoricalRange':
                result[hp_name] = CategoricalRange(
                    choices=hp_spec['choices'],
                )
        return result


def generate_benchmark_search_spaces(
    n_benchmarks: int,
    random_state: int = 42,
) -> Dict[int, Dict[str, Union[IntRange, FloatRange, CategoricalRange]]]:
    """Generate multiple search spaces for benchmarks.
    
    Args:
        n_benchmarks: Number of search spaces to generate
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping benchmark IDs to search spaces
    """
    generator = SearchSpaceGenerator(random_state=random_state)
    search_spaces = {}
    
    for benchmark_id in range(n_benchmarks):
        # Use different seed for each benchmark
        generator.random_state = random_state + benchmark_id
        random.seed(random_state + benchmark_id)
        np.random.seed(random_state + benchmark_id)
        
        search_space = generator.generate_search_space()
        search_spaces[benchmark_id] = search_space
        
        logger.info(
            f"Generated search space for benchmark {benchmark_id} with "
            f"{len(search_space)} hyperparameters"
        )
    
    return search_spaces
