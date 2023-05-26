# qoc-experiments
QOC experiments using the Pico.jl family of packages

## Setup
1. Install Julia using, e.g., [jill.py](https://github.com/johnnychen94/jill.py) and follow the instructions there to install Julia 1.8.5 (or later).
    
2. Clone this repository and `cd` into it.

3. Instantiate the Julia environment by entering the julia REPL with `julia --project=.`

4. then in the REPL, enter the package manager by typing `]` and then instantiate the environment by typing `instantiate`: 

    ```julia
    julia> ]
    (qoc-experiments) pkg> instantiate
    ```

## QRAM

to run the pi gate QRAM experiment from the terminal, in the repo directory, type 

```shell
julia --project=. experiments/QRAM/pi_gate.jl
```

to multithread the experiment, use the `-t` flag:

```shell
julia -t <n-threads> --project=. experiments/QRAM/pi_gate.jl
```

