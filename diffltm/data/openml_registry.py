from diffltm.data.dataset_spec import OpenMLDatasetSpec

ADULT = OpenMLDatasetSpec(
    name="adult",
    openml_id=45068,
    target="income",
    task="binclass",
    n_classes=2,
    numeric_cols=[
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ],
    categorical_cols=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "income",
    ],
    missing_tokens=["?"],
    description={
        "age": "Age of individual",
        "fnlwgt": "Final sampling weight",
        "education-num": "Years of education",
        "capital-gain": "Capital gains",
        "capital-loss": "Capital losses",
        "hours-per-week": "Hours worked per week",
        "workclass": "Type of employer",
        "education": "Highest education level",
        "marital-status": "Marital status",
        "occupation": "Type of occupation",
        "relationship": "Relationship in household",
        "race": "Race",
        "sex": "Sex",
        "native-country": "Country of origin",
        "income": "Income >50K (target)",
    },
)

OPENML_REGISTRY = {"adult": ADULT}
