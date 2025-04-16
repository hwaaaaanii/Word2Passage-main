# config.py
IMPORTANCE_CONFIG = {
    "nq": {
        "description": (0.2, 1.2, 1.6),
        "entity": (0.6, 0.8, 1.2),
        "person": (1.6, 1.2, 0.4),
        "numeric": (1.6, 1.6, 0.2),
        "location": (1.2, 1.4, 0.8),
    },
    "hotpotqa": {
        "description": (1.4, 0.6, 1.0),
        "entity": (0.4, 1.0, 1.2),
        "person": (0.8, 1.6, 0.6),
        "numeric": (1.4, 1.4, 1.2),
        "location": (1.6, 1.2, 0.8),
    },
    "squad": {
        "description": (1.0, 0.8, 1.6),
        "entity": (0.4, 0.6, 1.0),
        "person": (1.4, 0.6, 1.4),
        "numeric": (0.4, 1.6, 1.2),
        "location": (0.6, 1.4, 0.8),
    },
    "msmarco": {
        "description": (0.2, 0.6, 1.6),
        "entity": (1.2, 0.8, 0.4),
        "person": (0.8, 1.4, 0.8),
        "numeric": (1.6, 1.4, 1.4),
        "location": (1.2, 1.6, 0.2),
    },
    "scifact": {
        "description": (1.2, 0.4, 0.2),
        "entity": (0.2, 0.2, 0.2),
        "person": (1.0, 1.0, 1.0),
        "numeric": (0.2, 0.8, 0.8),
        "location": (1.0, 1.0, 1.0),
    },
    "fiqa": {
        "description": (0.4, 0.6, 0.4),
        "entity": (0.6, 1.4, 0.2),
        "person": (1.2, 1.4, 0.2),
        "numeric": (1.2, 1.2, 1.2),
        "location": (0.8, 0.2, 0.4),
    },
    "nfcorpus": {
        "description": (0.4, 0.2, 1.2),
        "entity": (0.4, 0.4, 0.4),
        "person": (0.8, 0.6, 0.4),
        "numeric": (0.4, 0.6, 0.2),
        "location": (1.0, 1.0, 1.0),
    },
    "trivia": {
        "description": (1.6, 0.8, 1.2),
        "entity": (0.8, 1.4, 0.2),
        "person": (1.6, 1.2, 1.0),
        "numeric": (0.6, 0.8, 1.6),
        "location": (0.8, 1.0, 0.4),
    }
}


WORD_COUNT = {
    "msmarco": 42.07,
    "trec-covid": 95.85,
    "webis-touche2020": 153.10,
    "scifact": 128.22,
    "nfcorpus": 138.80,
    "arguana": 111.78,
    "scidocs": 109.66,
    "nq": 56.01,
    "hotpotqa": 35.15,
    "fiqa": 89.50,
    "squad": 73.50,
    "trivia": 73.50,
}