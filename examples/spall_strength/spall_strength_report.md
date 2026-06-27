# Spall Strength Physics-Informed GPR Report

## Reference

Sahu, H. et al. **Machine-learning based prediction of spall strengths of metals and alloys.** Journal of Applied Physics accepted manuscript, DOI: 10.1063/5.0248560.

## Dataset

- File: `dataset.csv`.
- Target: median spall strength `spall_median`.
- Features: UTS, tensile yield strength, bulk modulus, Young's modulus, density, hardness, and fracture toughness.

## Physics Basis

Spall resistance depends on material compressibility, plastic flow strength, and failure resistance. The paper discusses Grady-style theoretical spall models for brittle and ductile response.

## Models Tested

- `Standard GPR`: seven-feature mechanical-property baseline.
- `PI-GPR: strength x bulk`: mean based on the square-root combination of yield strength and bulk modulus.
- `PI-GPR: Grady ductile proxy`: mean based on a ductile spall proxy.
- `PI-GPR: Grady brittle proxy`: mean based on a fracture-toughness and bulk-modulus proxy.
- `PI-GPR: ductile/brittle switch`: uses fracture toughness to switch between ductile and brittle proxies.

## Physics Mean Functions

### Strength x Bulk

```text
mean = baseline + scale * sqrt(K * Y)
```

- Features: bulk modulus `K`, tensile yield strength `Y`.
- Learned parameters: `baseline`, `scale`.

### Grady Ductile Proxy

```text
mean = baseline + scale * sqrt(2 * K * Y * epsilon_c)
```

- Features: bulk modulus `K`, tensile yield strength `Y`.
- Fixed parameter: `epsilon_c = 0.15`.
- Learned parameters: `baseline`, `scale`.

### Grady Brittle Proxy

```text
mean = baseline + scale * (K * KIC^2)^(1/3)
```

- Features: bulk modulus `K`, fracture toughness `KIC`.
- Learned parameters: `baseline`, `scale`.

### Ductile/Brittle Switch

```text
if KIC >= 40 MPa sqrt(m): use ductile proxy
else: use brittle proxy
```

- Features: bulk modulus, tensile yield strength, fracture toughness.
- Fixed threshold: `KIC = 40 MPa sqrt(m)`.
- Learned parameters: `baseline`, `scale`.

## Learning Curve Protocol

- Training pool/test split is repeated 20 times.
- Training pool usage is varied from 10 percent to 100 percent in 10 percent steps.
- Each point reports mean and standard deviation across the 20 splits.
- Metrics: RMSE, R2, MAE, and Pearson r.

## Notes

- These physics models are intentionally proxies rather than final mechanistic equations.
- The goal is to benchmark several interpretable mean functions before selecting the top models.
