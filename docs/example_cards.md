# Example Dataset And Model Cards

Dataset cards and model cards document what each public example contains, how
it is validated, and where its assumptions are likely to fail. They are meant
to make the examples easier to audit before users adapt them to new materials
datasets.

## Public Example Cards

| Example | Dataset card | Model card | Main notebook |
| --- | --- | --- | --- |
| OPV PCE | [dataset card](https://github.com/harikrishna-chem/matgpr/blob/main/examples/opv/dataset_card.md) | [model card](https://github.com/harikrishna-chem/matgpr/blob/main/examples/opv/model_card.md) | `examples/opv/opv_gpr_modeling.ipynb` |
| Solvent diffusivity | [dataset card](https://github.com/harikrishna-chem/matgpr/blob/main/examples/solvent_diffusivity/dataset_card.md) | [model card](https://github.com/harikrishna-chem/matgpr/blob/main/examples/solvent_diffusivity/model_card.md) | `examples/solvent_diffusivity/solvent_diffusivity_gpr_modeling.ipynb` |

## What The Cards Record

Each dataset card records:

- source paper and citation,
- dataset file and notebook location,
- target definition and units,
- feature columns and featurization logic,
- cleaning and filtering rules,
- validation protocol,
- intended use and known limitations.

Each model card records:

- model family and implementation,
- standard and physics-informed mean functions,
- equations used to introduce physics,
- learned physics parameters and GP hyperparameters,
- validation summary,
- intended use and limitations.

## Why This Matters

Physics-informed GPR examples are easiest to trust when users can see exactly
which features enter the physics equation, which parameters are learned during
GP training, how the model was validated, and where the assumptions are likely
to break. These cards make those details explicit without forcing users to
read the full notebook first.

