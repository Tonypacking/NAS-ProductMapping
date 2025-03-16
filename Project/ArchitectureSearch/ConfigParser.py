import configparser
import itertools

class NeatConfigParser:
    @staticmethod
    def _getDefaultAttributes():
        """_summary_
        Default config values
        Returns:
            _type_: _description_
        """
        return {
            ("NEAT", 'fitness_criterion'): 'mean',
            ("NEAT", 'no_fitness_termination'): True,
            ("NEAT", 'fitness_threshold'): 950,
            ("NEAT", 'pop_size'): 150,
            ("NEAT", 'reset_on_extinction'): True,
            
            ("DefaultStagnation", 'species_fitness_func'): 'max',
            ("DefaultStagnation", 'max_stagnation'): 6,
            ("DefaultStagnation", 'species_elitism'): 2,

            ("DefaultReproduction", 'elitism'): 2,
            ("DefaultReproduction", 'survival_threshold'): 0.35,

            ("DefaultGenome", 'activation_default'): 'sigmoid',
            ("DefaultGenome", 'activation_mutate_rate'): 0.01,
            ("DefaultGenome", 'activation_options'): 'sigmoid',

            ("DefaultGenome", 'aggregation_default'): 'sum',
            ("DefaultGenome", 'aggregation_mutate_rate'): 0.0,
            ("DefaultGenome", 'aggregation_options'): 'sum',

            ("DefaultGenome", 'bias_init_mean'): 0.0,
            ("DefaultGenome", 'bias_init_stdev'): 1.0,
            ("DefaultGenome", 'bias_max_value'): 30.0,
            ("DefaultGenome", 'bias_min_value'): -30.0,
            ("DefaultGenome", 'bias_mutate_power'): 0.5,
            ("DefaultGenome", 'bias_mutate_rate'): 0.7,
            ("DefaultGenome", 'bias_replace_rate'): 0.1,

            ("DefaultGenome", 'compatibility_disjoint_coefficient'): 1.0,
            ("DefaultGenome", 'compatibility_weight_coefficient'): 0.5,

            ("DefaultGenome", 'conn_add_prob'): 0.9,
            ("DefaultGenome", 'conn_delete_prob'): 0.3,

            ("DefaultGenome", 'enabled_default'): True,
            ("DefaultGenome", 'enabled_mutate_rate'): 0.5,

            ("DefaultGenome", 'feed_forward'): True,
            ("DefaultGenome", 'initial_connection'): 'full',

            ("DefaultGenome", 'node_add_prob'): 0.9,
            ("DefaultGenome", 'node_delete_prob'): 0.3,

            ("DefaultGenome", 'num_hidden'): 0,
            ("DefaultGenome", 'num_inputs'): 34,
            ("DefaultGenome", 'num_outputs'): 1,

            ("DefaultGenome", 'response_init_mean'): 1.0,
            ("DefaultGenome", 'response_init_stdev'): 0.0,
            ("DefaultGenome", 'response_max_value'): 30.0,
            ("DefaultGenome", 'response_min_value'): -30.0,
            ("DefaultGenome", 'response_mutate_power'): 0.0,
            ("DefaultGenome", 'response_mutate_rate'): 0.0,
            ("DefaultGenome", 'response_replace_rate'): 0.0,

            ("DefaultGenome", 'weight_init_mean'): 0.0,
            ("DefaultGenome", 'weight_init_stdev'): 1.0,
            ("DefaultGenome", 'weight_max_value'): 300,
            ("DefaultGenome", 'weight_min_value'): -300,
            ("DefaultGenome", 'weight_mutate_power'): 0.5,
            ("DefaultGenome", 'weight_mutate_rate'): 0.8,
            ("DefaultGenome", 'weight_replace_rate'): 0.1,

            ("DefaultSpeciesSet", 'compatibility_threshold'): 3.0
            }

    def __init__(self, directory: str):
        self._dir = directory
        self._suffix = '.ini'
    
    def _add_default_values(self, parser: configparser.ConfigParser):
        """_summary_
        Adds default neat config values.
        Args:
            parser (configparser.ConfigParser): _description_
        """
        default = NeatConfigParser._getDefaultAttributes()
        for (section, options), value in zip(default.keys(), default.values()):
            if not parser.has_section(section):
                parser.add_section(section)

            if not parser.has_option(section,options):
                parser.set(section, options, str(value))


    def createConfig(self, config: dict[tuple[str, str], list[any]], add_default_values: bool = True):
        """_summary_
        Creates neat config with all possible value combinations specified in the dictionary. Works similary as a grid search in sklearn.
        Args:
            config (dict[tuple[str, str], list[any]]): dictionary where key is tuple of section and option and whose value is a list of all possible values.
            add_default_values (bool, optional): Defaults to True. If true adds in default values in config file otherwise not.
        """
        combinations = list(itertools.product(*config.values()))
        
        for comb in combinations:
            name= self._dir+"/"
            parser = configparser.ConfigParser()

            for (section, option), value in zip(config.keys(), comb): # keys should be in the same order as the index in combinationn
                if not parser.has_section(section=section):
                    parser.add_section(section)
                parser.set(section, option, str(value))
                name += f"{option}-{value} "
            name += self._suffix

            if add_default_values:
                self._add_default_values(parser=parser)

            with open(name, 'w') as f:
                parser.write(f)
