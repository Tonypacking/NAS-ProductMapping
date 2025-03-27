# Finding neuron architectures in product mapping.

## launch program
With default arguments: `python3 main`.
Requires input/input.json (if input argument isn't set) which validates json schema. And it also requires data with all Product mapping datasets from [this dataset](https://github.com/kackamac/Product-Mapping-Datasets)
### Program arguments
- **Neat arguments**
    - **Parallel**
        - If set, genome evaluation runs in parralel.
        - Example usage: `python3 main.py  --parallel` or `python3  python3 main.py  --par`
        - Default: False 
        - Type: bool
    - **Iterations**
        -  Sets maximum number of generations in each neat generation.
        - Example usage: `python3 main.py --iterations 50` or `python3 main.py  --iter 50`
        - Default: 50
        - Type: integer

- **Dataset preprocessing arguments**
    - **Dimenstion reduction**
        - If set dataset is preprocessed and dimensions are reduced by  sklearn library either with LDA or PCA.
        - Example usage `python3 main.py dimension_reduction raw` or `python3 main.py  dims pca`
        - Default: raw
        - Type: string
        - Possible values: raw, pca, lda.

    - **Data standardization**
        - If set, standardizes dataset by sklearn library with StandardScaler.
        - Example usage: `python3 main.py --scale ` or `python3 main.py --s`
        - Default: False
        - Type: bool

- **Dataset arguments**
    - **Dataset**
        - Runs neruon architecture search on passed dataset.
        - Possible dataset names: 
            - google
            - walmart
            - promapcz
            - promapen
            - promapczext
            - promapenext
            - amazonext
        - Example usage: `python3 main.py --dataset promapcz` or `python3 main.py --o promapczext`
        - Default: promapcz
        - Type string, only which are possible others are ignored and program won't start.
        
- **Output arguments**
    - **Output path**
        - Creates if directory doesn't exist with a passed path where the output of this program is stored.
        - Example usage `python3 main.py --output output` or `python3 main.py --o output `
        - Default: Output
        - Type: Any valid path
    - **Data validations**
        - If set, model is validated against all possible target dataset. If number of features  doesn't match it is ignored. If dimensions are reduces, it first trires to transforms test dataset if it fails it is ignored.
        - Example usage: `python3 main.py --validate_all` or `python3 main.py --v`
        - Default: Validations only with designated test set
        - Type: Bool
    - **kbest**
        -  Saves results of top k networks in seperate folder.
        - Example usage `python3 main.py --kbest` or `python3 main.py --k`
        - Default: 10
        - Type: integer 

- **Config generation**
    -   **Configuration directory**
        - Directory name where the generation neat configs are to be generated. 
        - Example usage: `python3 main.py --config_directory path` or `python3 main.py --dir config/generation/example/path`
        - Default: ConfigGeneration
        - Type: Any valid path for a directory. 

    -   **Configuration generation**
        - If set it disables neat config generation.
        - Example usage: `python3 main.py --config_generation` or `python3 main.py --g`
        - Default: True
        - Type: bool

    -   **Input**
        - Path to input json file. Needs to be valid against json schema.
        - Example usage: `python3 main.py --input input/input.json` or `python3 main.py --i input/input.json`
        - Default: input/input.json
        - Type: any valid path to json file which is valid against json schema.

    -   **Default**
        - If set default value generations in genrating neat config file are disabled.
        - Example usage: `python3 main.py --default` or `python3 main.py --def`
        - Default: False
        -  Type: bool


    -   **Search for all files in configuration directory**
        - If set instead of searching architectures only for newly generated **.ini** files. It tris to find architectures of every**.init** files in the ConfigGeneration directory (if default arguments isn't changed otherwises it looks for all .ini files in the set directory).
        - Example usage: `python3 main.py --all_files` or `python3 main.py --all`
        - Default: False
        - Type: bool


