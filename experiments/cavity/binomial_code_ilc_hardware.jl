using QuantumCollocation
using NamedTrajectories
using IterativeLearningControl
using PyCall

@pyinclude joinpath(@__DIR__, "run_experiment_optimize_loop_binom.py")

ilc_max_iter = 5
samples = 1000

function g(
    A::Matrix{Float64},
    times::AbstractVector{Float64},
    τs::AbstractVector{Int};
    acquisition_num::Int=samples
)
    as = collect(eachcol(A))
    ys = py"take_controls_and_measure"(times, as, τs, acq_num=acquisition_num)
    ys = collect(eachcol(transpose(ys)))
    return ys
end

cavity_levels = 14

function g_ref(x)
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

ydim = g_pop(vcat(traj[end].ψ̃1, traj[end].ψ̃2))

τs = [traj.T]

experiment = QuantumHardwareExperiment(g, ydim, τs)

α = 1.0
β = 0.0
optimizer = BacktrackingLineSearch(J; α=α, β=β, verbose=true, global_min=true)

prob = ILCProblem(
    traj,
    system,
    integrators,
    experiment,
    optimizer;
    state_names=[:ψ̃1, :ψ̃2],
)

experiment_name = "samples_$(samples)"

output_dir = @__DIR__() * "//plots//binomial_code//hardware"

io = open(generate_file_path(
    "txt",
    experiment_name * "_log",
    output_dir
), "w")

solve!(prob; io=io)

plot_path = generate_file_path(
    "gif",
    experiment_name * "_objective",
    output_dir
)

save_animation(plot_path, prob)

cos_sims_plot_path = generate_file_path(
    "png",
    experiment_name * "_cos_sims",
    output_dir
)

problem_save_path = generate_file_path(
    "jld2",
    experiment_name * "_problem",
    output_dir
)

jldsave(problem_save_path; prob)

dZs = [dZ.datavec for dZ ∈ prob.dZs]

plot_cosine_similarity(cos_sims_plot_path, dZs)
