using NamedTrajectories
using QuantumCollocation
import QuantumCollocation: lift
using QuantumIterativeLearningControl
using LinearAlgebra
using JLD2
using PyCall

experimental_levels = 3

qubits = 4

ilc_max_iter = 10

@pyinclude "experiments/QRAM/pi_gate.py"

data_path = joinpath(
    @__DIR__,
    # "data_demo/unitary_levels_3_T_100_dt_2.054693946482075_dda_0.0001_a_0.02_max_iter_5000_00000.jld2"
    "data/levels_2_2_2_2_drives_1_2_3_T_100_dt_2.0_dda_0.0001_a_0.02_max_iter_500_00001.jld2"
)

data = load_problem(data_path; return_data=true)
traj = data["trajectory"]
system = data["system"]
integrators = data["integrators"]

α = [225.78, 100.33, 189.32, 172.15] * 1e-3 # GHz

χ = Symmetric([
    0 -5.10982939 -0.18457118 -0.50235316;
    0       0     -0.94914758 -1.07618574;
    0       0           0     -0.44607489;
    0       0           0           0
]) * 1e-3 # GHz

â_dag = create(experimental_levels)
â = annihilate(experimental_levels)

lift(op, i; l=experimental_levels) = lift(op, i, qubits; l=l)
# lift_gate_qubit(op)

# drift hamiltonian for ith qubit
H_q(i; l=experimental_levels) = -α[i] / 2 * lift(â_dag, i; l=l)^2 * lift(â, i; l=l)^2

# drift interaction hamiltonian for ith and jth qubit
H_c_ij(i, j; l=experimental_levels) =
    χ[i, j] *
    lift(â_dag, i; l=l) *
    lift(â, i; l=l) *
    lift(â_dag, j; l=l) *
    lift(â, j; l=l)

# drive hamiltonian for ith qubit, real part
H_d_real(i; l=experimental_levels) = 1 / 2 * (lift(â_dag, i; l=l) + lift(â, i; l=l))

# drive hamiltonian for ith qubit, imaginary part
H_d_imag(i; l=experimental_levels) = 1im / 2 * (lift(â_dag, i; l=l) - lift(â, i; l=l))

# total drift hamiltonian
H_drift =
    sum(H_q(i) for i = 1:qubits) +
    sum(H_c_ij(i, j) for i = 1:qubits, j = 1:qubits if j > i)

H_drift *= 2π

# make vector of drive hamiltonians: [H_d_real(1), H_d_imag(1), H_d_real(2), ...]
# there's probably a cleaner way to do this lol
# H_drives = collect.(vec(vcat(
#     transpose(Matrix{ComplexF64}.([H_d_real(i) for i = 1:qubits])),
#     transpose(Matrix{ComplexF64}.([H_d_imag(i) for i = 1:qubits]))
# )))

H_drives = Matrix{ComplexF64}.([
    H_d_real(1), H_d_imag(1),
    H_d_real(2), H_d_imag(2),
    H_d_real(3), H_d_imag(3),
    H_d_real(4), H_d_imag(4),
])
H_drives .*= 2π

# make quantum system
experimental_system = QuantumSystem(H_drift, H_drives)

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
    state_names=fill(:Ũ⃗, length(ψ_inits_experiment)),
    # state_names=[:ψ̃1, :ψ̃2, :ψ̃3, :ψ̃4, :ψ̃5, :ψ̃6, :ψ̃7, :ψ̃8],
    # state_names=[:ψ̃1, :ψ̃1, :ψ̃1, :ψ̃2, :ψ̃2, :ψ̃2],
    R=R
)

solve!(prob; project=true)

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

# g_experiment(prob.Zs[end].a, times(prob.Zs[end]), [traj.T]; print_result=true)
