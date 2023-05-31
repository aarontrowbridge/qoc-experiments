using QuantumCollocation
using NamedTrajectories
using IterativeLearningControl

cavity_levels = 14

function g_pop(x)
    y = []
    x1 = x[1:(2*3*14)]
    x2 = x[(2*3*14) + 1:(4*3*14)]
    append!(y, sum(x1[1:cavity_levels].^2 + x1[3*cavity_levels .+ (1:cavity_levels)].^2))
    append!(
        y,
        sum(
            x1[cavity_levels .+ (1:cavity_levels)].^2 +
            x1[4*cavity_levels .+ (1:cavity_levels)].^2
        )
    )
    for i = 1:10
        append!(y,
            x1[i]^2 +
            x1[i + 3 * cavity_levels]^2 +
            x1[i + cavity_levels]^2 +
            x1[i + 4 * cavity_levels]^2 +
            x1[i + 2 * cavity_levels]^2 +
            x1[i + 5 * cavity_levels]^2
        )
        #append!(y, x[i + cavity_levels]^2 + x[i+3*cavity_levels]^2)
    end
    append!(y, sum(x2[1:cavity_levels].^2 + x2[3*cavity_levels .+ (1:cavity_levels)].^2))
    append!(
        y,
        sum(
            x2[cavity_levels .+ (1:cavity_levels)].^2 +
            x2[4*cavity_levels .+ (1:cavity_levels)].^2
        )
    )
    for i = 1:10
        append!(y,
            x2[i]^2 +
            x2[i + 3 * cavity_levels]^2 +
            x2[i + cavity_levels]^2 +
            x2[i + 4 * cavity_levels]^2 +
            x2[i + 2 * cavity_levels]^2 +
            x2[i + 5 * cavity_levels]^2
        )
        #append!(y, x[i + cavity_levels]^2 + x[i+3*cavity_levels]^2)
    end
    return convert(typeof(x), y)
end

data_path = joinpath(@__DIR__, "data/binomial_code/transmon_3_T_200_dt_15.0_Q_200.0_R_L1_10.0_max_iter_5000_dda_bound_1.0e-5_00000.jld2")

data = load_problem(data_path; return_data=true)

traj = data["trajectory"]
system = data["system"]
integrators = data["integrators"]

experiment
