using NamedTrajectories
using QuantumCollocation
import QuantumCollocation: lift
using QuantumIterativeLearningControl
using LinearAlgebra
using JLD2

include(joinpath(@__DIR__, "qram_system.jl"))

experimental_levels = 3

qubits = [1, 2, 3, 4]
levels = [3, 3, 3, 3]
drives = [1, 2, 3, 4]

experimental_system = QRAMSystem(; qubits=qubits, levels=levels, drives=drives)

data_path = joinpath(
    @__DIR__,
    "data/levels_2_2_2_2_drives_1_2_3_4_T_100_dt_2.0_dda_0.0001_a_0.02_max_iter_500_00000.jld2"
)


ilc_max_iter = 5


data = load_problem(data_path; return_data=true)
traj = data["trajectory"]
system = data["system"]
integrators = data["integrators"]

ref_levels = 2

# create goal unitary
g = cavity_state(0, ref_levels)
e = cavity_state(1, ref_levels)
eg = e * g'
ge = g * e'
ee = e * e'
gg = g * g'
Id = ee + gg
U_goal = Id ⊗ (eg + ge) ⊗ Id ⊗ Id

ψ_inits_ref = [
    g ⊗ g ⊗ g ⊗ g,
    g ⊗ g ⊗ g ⊗ e,
    g ⊗ g ⊗ e ⊗ g,
    g ⊗ e ⊗ g ⊗ g,
    g ⊗ e ⊗ e ⊗ g,
    g ⊗ e ⊗ g ⊗ e,
    # g ⊗ (e + g) / √2 ,
    # g ⊗ (e + im * g) / √2,
    # e ⊗ g,
    # e ⊗ e,
    # e ⊗ (e + g) / √2,
    # e ⊗ (e + im * g) / √2,
    # (-e ⊗ e + im * e ⊗ g + g ⊗ g + im * g ⊗ e) / 2,
]


ψ_goals_ref = [U_goal * ψ_init for ψ_init ∈ ψ_inits_ref]




# g_refs = Function[ψ̃ -> [fidelity(iso_to_ket(ψ̃), ψ_goal)] for ψ_goal ∈ ψ_goals_ref]
g_refs = Function[Ũ⃗ -> [fidelity(iso_vec_to_operator(Ũ⃗) * ψ_init, ψ_goal)] for (ψ_init, ψ_goal) ∈ zip(ψ_inits_ref, ψ_goals_ref)]

# ψ̃_finals = [traj[end].ψ̃1, traj[end].ψ̃2, traj[end].ψ̃3, traj[end].ψ̃4, traj[end].ψ̃5, traj[end].ψ̃6, traj[end].ψ̃7, traj[end].ψ̃8]
# ψ̃_finals = [
#     traj[end].ψ̃1,
#     traj[end].ψ̃1,
#     traj[end].ψ̃1,
#     traj[end].ψ̃2,
#     traj[end].ψ̃2,
#     traj[end].ψ̃2,
# ]

Ũ⃗_final = traj[end].Ũ⃗

y_goal = vcat([g_ref(Ũ⃗_final) for g_ref ∈ g_refs]...)

println("y_goal = ")
display(y_goal)



# ψ_inits_experiment = [ψ ⊗ g ⊗ g for ψ ∈ ψ_inits_ref]

g = cavity_state(0, experimental_levels)
e = cavity_state(1, experimental_levels)

ψ_inits_experiment = [
    g ⊗ g ⊗ g ⊗ g,
    g ⊗ g ⊗ g ⊗ e,
    g ⊗ g ⊗ e ⊗ g,
    g ⊗ e ⊗ g ⊗ g,
    g ⊗ e ⊗ g ⊗ e,
    g ⊗ e ⊗ e ⊗ g,
]



ψ̃_inits_experiment = ket_to_iso.(ψ_inits_experiment)

eg = e * g'
ge = g * e'
ee = e * e'
gg = g * g'

Id = ee + gg

U_goal = Id ⊗ (eg + ge) ⊗ Id ⊗ Id

ψ_goals_experiment = [U_goal * ψ₀ for ψ₀ ∈ ψ_inits_experiment]

τs = [traj.T]

gs_fidelity = Function[
    ψ̃ -> [fidelity(iso_to_ket(ψ̃), ψ_goal)]
        for ψ_goal ∈ ψ_goals_experiment
]

# gs_full_state = Function[ψ̃ -> ψ̃ for ψ_goal ∈ ψ_goals]

experiment_fidelity =
    QuantumSimulationExperiment(experimental_system, ψ̃_inits_experiment, gs_fidelity, τs)

# experiment_full_state =
#     QuantumSimulationExperiment(experimental_system, ψ̃_inits, gs_full_state, τs)

# g_experiment(traj.a, times(traj), [traj.T]; print_result=true)

y_initial = experiment_fidelity(traj.a, timesteps(traj)).ys[end]
println("y_initial = ", y_initial)

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
    # ψ̃1=R_val,
    # ψ̃2=R_val,
    # ψ̃3=R_val,
    # ψ̃4=R_val,
    # ψ̃5=R_val,
    # ψ̃6=R_val,
    # ψ̃7=R_val,
    # ψ̃8=R_val,
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
    # state_names=[:ψ̃1, :ψ̃2, :ψ̃3, :ψ̃4, :ψ̃5, :ψ̃6, :ψ̃7, :ψ̃8],
    # state_names=[:ψ̃1, :ψ̃1, :ψ̃1, :ψ̃2, :ψ̃2, :ψ̃2],
    R=R,
    QP_verbose=true,
    objective_hessian=false,
)

solve!(prob; manifold_projection=false)

experiment_name = "unitary_pi_gate_ilc_R_val_$(R_val)"

output_dir = joinpath(@__DIR__, "plots_ILC/pi_gate")

plot_path = generate_file_path(
    "gif",
    experiment_name * "_objective",
    output_dir
)

save_animation(plot_path, prob; plot_states=false)

cos_sims_plot_path = generate_file_path(
    "png",
    experiment_name * "_cos_sims",
    output_dir
)

save_dir = joinpath(@__DIR__, "data_ILC/pi_gate")

problem_save_path = generate_file_path(
    "jld2",
    experiment_name * "_problem",
    save_dir
)

save(problem_save_path, "prob", prob)

dZs = [dZ.datavec for dZ ∈ prob.dZs]

plot_cosine_similarity(cos_sims_plot_path, dZs)

y_final = experiment_fidelity(prob.Zs[end].a, timesteps(prob.Zs[end])).ys[end]
println("y_initial = ", y_initial)
println("y_final =   ", y_final)

ilc_traj_path = generate_file_path(
    "jld2",
    experiment_name * "_final_traj",
    save_dir
)

save(ilc_traj_path, "traj", prob.Zs[end])
