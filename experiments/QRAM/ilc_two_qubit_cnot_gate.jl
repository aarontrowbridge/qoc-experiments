using NamedTrajectories
using QuantumCollocation
import QuantumCollocation: lift
using QuantumIterativeLearningControl
using LinearAlgebra
using JLD2

include(joinpath(@__DIR__, "qram_system.jl"))

experimental_levels = 3

qubits = [1, 2]
levels = fill(experimental_levels, length(qubits))
drives = [1, 2]

experimental_system = QRAMSystem(; qubits=qubits, levels=levels, drives=drives)

data_path = joinpath(@__DIR__, "data/gate_CX_levels_2_2_drives_1_2_T_100_dt_2.0_dda_0.001_a_0.02_max_iter_5000_00000.jld2")


ilc_max_iter = 5


data = load_problem(data_path; return_data=true)
traj = data["trajectory"]
system = data["system"]
integrators = data["integrators"]

U_goal = iso_vec_to_operator(traj.goal.Ũ⃗)

ref_levels = 2
g = cavity_state(0, ref_levels)
e = cavity_state(1, ref_levels)

ψ_inits_ref = [
    g ⊗ e,
    e ⊗ g,
    g ⊗ g,
    e ⊗ e,
]

ψ_goals_ref = [U_goal * ψ_init for ψ_init ∈ ψ_inits_ref]




g_refs = Function[
    Ũ⃗ -> [fidelity(iso_vec_to_operator(Ũ⃗) * ψ_init, ψ_goal)]
        for (ψ_init, ψ_goal) ∈ zip(ψ_inits_ref, ψ_goals_ref)
]


Ũ⃗_final = traj[end].Ũ⃗

y_goal = vcat([g_ref(Ũ⃗_final) for g_ref ∈ g_refs]...)

println("y_goal = ", y_goal)

g = cavity_state(0, experimental_levels)
e = cavity_state(1, experimental_levels)

ψ_inits_experiment = [
    g ⊗ g,
    g ⊗ e,
    e ⊗ g,
    e ⊗ e,
]

ψ̃_inits_experiment = ket_to_iso.(ψ_inits_experiment)

eg = e * g'
ge = g * e'
ee = e * e'
gg = g * g'

Id = ee + gg

U_goal = Id ⊗ (eg + ge) ⊗ Id ⊗ Id

ψ_goals_experiment = [
    g ⊗ g,
    g ⊗ e,
    e ⊗ e,
    e ⊗ g,
]

τs = [traj.T]

gs_fidelity = Function[
    ψ̃ -> [fidelity(iso_to_ket(ψ̃), ψ_goal)]
        for ψ_goal ∈ ψ_goals_experiment
]

experiment_fidelity =
    QuantumSimulationExperiment(experimental_system, ψ̃_inits_experiment, gs_fidelity, τs)

y_initial = experiment_fidelity(traj.a, timesteps(traj)).ys[end]
println("y_init = ", y_initial)

J(y) = norm(y - y_goal, 1)

α = 1.0
β = 0.0
optimizer = BacktrackingLineSearch(J; α=α, β=β, verbose=true, global_min=false)

R_val = 0.05

R = (
    a=R_val,
    da=R_val,
    dda=R_val,
    Ũ⃗=R_val,
)

prob = ILCProblem(
    traj,
    system,
    integrators,
    experiment_fidelity,
    optimizer;
    g_ref=g_refs,
    max_iter=ilc_max_iter,
    quantum_state_names=fill(:Ũ⃗, length(ψ_inits_experiment)),
    R=R,
    QP_verbose=true,
    objective_hessian=false,
    y_final=fill(1.0, length(ψ_inits_experiment)),
)

solve!(prob; manifold_projection=false)

experiment_name = "two_transmon_cnot_gate_ilc_R_val_$(R_val)"

output_dir = joinpath(@__DIR__, "plots_ILC/pi_gate")

plot_path = generate_file_path(
    "gif",
    experiment_name * "_objective",
    output_dir
)

save_animation(plot_path, prob; plot_states=false)

save_dir = joinpath(@__DIR__, "data_ILC/pi_gate")

problem_save_path = generate_file_path(
    "jld2",
    experiment_name * "_problem",
    save_dir
)

save(problem_save_path, "prob", prob)

y_final = experiment_fidelity(prob.Zs[end].a, timesteps(prob.Zs[end])).ys[end]
println("y_initial = ", y_initial)
println("y_final =   ", y_final)

ilc_traj_path = generate_file_path(
    "jld2",
    experiment_name * "_final_traj",
    save_dir
)

save(ilc_traj_path, "traj", prob.Zs[end])
