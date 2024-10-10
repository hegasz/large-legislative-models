"""
These hold the configurations for each experiment.
To change the hyperparameters for an experiment in a gridsearch, edit them here.
"""

configs = {
    "llm_harvest": {
        "base": {
            "principal": "LLM",
            "env_name": "commons_harvest__open"
        },
        "hparams": {
            "method": ["llm_harvest"],
            "principal_lr": [1e-4, 1e-3],
            "seed": [1,2,3,4,5,6,7,8,9,10],
        },
    },
    "llm_cleanup": {
        "base": {
            "principal": "LLM",
            "env_name": "clean_up"
        },
        "hparams": {
            "method": ["llm_harvest"],
            "principal_lr": [1e-3, 5e-4, 1e-3],
            "seed": [1,2,3,4,5,6,7,8,9,10],
        },
    },
}
