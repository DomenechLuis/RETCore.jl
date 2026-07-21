# RETCore.jl

RETCore.jl provides statistical tools for the analysis and interpretation of Rain Erosion Test (RET) results.

## Installation

```julia
using Pkg

Pkg.add(url="https://github.com/usuario/RETCore.jl")
```

## Example

```julia
using RETCore

data = DataObject(v, N)

fo = FitObject(
    normal_linear_model,
    data
)

prepare_data!(fo)
fit!(fo)
```

## Disclaimer

This software is intended for research purposes only.

The software is provided "as is" and without warranty of any kind. No guarantee is made regarding the correctness, accuracy, reliability, completeness, or suitability of the software or any results obtained from its use.

## Author

Luis Doménech Ballester