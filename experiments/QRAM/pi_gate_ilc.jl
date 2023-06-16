using NamedTrajectories
using QuantumCollocation
import QuantumCollocation: lift
using IterativeLearningControl
using LinearAlgebra
using JLD2
using PyCall

experimental_levels = 3

qubits = 4

ilc_max_iter = 20

@pyinclude "experiments/QRAM/pi_gate.py"

data_path = joinpath(
    @__DIR__,
    "data/limited_drives_T_100_dt_1.0_dda_0.002_a_0.02_max_iter_500_00001.jld2"
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

H_drives = Matrix{ComplexF64}.([H_d_real(1), H_d_imag(1), H_d_real(2), H_d_imag(2)])
# H_drives = Matrix{ComplexF64}.([H_d_real(2), H_d_imag(2)])
H_drives .*= 2π

# make quantum system
experimental_system = QuantumSystem(H_drift, H_drives)

g = [1.0, 0.0]
e = [0.0, 1.0]
eegg = e ⊗ e ⊗ g ⊗ g
eggg = e ⊗ g ⊗ g ⊗ g
gggg = g ⊗ g ⊗ g ⊗ g
gegg = g ⊗ e ⊗ g ⊗ g

ψ_inits_ref = [
    (-eegg + im * eggg + gggg + im * gegg) / 2,
    eggg
]

Id = 1.0I(2)
eg = e * g'
ge = g * e'
U_goal = Id ⊗ (eg + ge) ⊗ Id ⊗ Id

ψ_goals_ref = [U_goal * ψ_init for ψ_init ∈ ψ_inits_ref]

function g_ref(Ũ⃗)
    U = iso_vec_to_operator(Ũ⃗)
    ψ_finals = [U * ψ_init for ψ_init ∈ ψ_inits_ref]
    # return [fidelity(ψ_final, ψ_goal) for (ψ_final, ψ_goal) ∈ zip(ψ_finals, ψ_goals_ref)]
    return vcat(ket_to_iso.(ψ_finals)...)
end

y_goal = g_ref(traj[end].Ũ⃗)

# println("y_goal = ", y_goal)

g = [1.0, 0.0, 0.0]
e = [0.0, 1.0, 0.0]
f = [0.0, 0.0, 1.0]
eegg = e ⊗ e ⊗ g ⊗ g
eggg = e ⊗ g ⊗ g ⊗ g
gggg = g ⊗ g ⊗ g ⊗ g
gegg = g ⊗ e ⊗ g ⊗ g

ψ_inits = [
    (-eegg + im * eggg + gggg + im * gegg) / 2,
    eggg
]

ψ̃_inits = ket_to_iso.(ψ_inits)

Id = 1.0I(experimental_levels)
eg = e * g'
ge = g * e'
ff = f * f'

U_goal = Id ⊗ (eg + ge + ff) ⊗ Id ⊗ Id

ψ_goals = [U_goal * ψ₀ for ψ₀ ∈ ψ_inits]

g_experiment(A, ts, τs; print_result=false) =
    [py"get_fidelities"(A, ts, print_result=print_result)]

τs = [traj.T]

gs_fidelity = Function[ψ̃ -> fidelity(iso_to_ket(ψ̃), ψ_goal) for ψ_goal ∈ ψ_goals]
gs_full_state = Function[ψ̃ -> ψ̃ for ψ_goal ∈ ψ_goals]


# experiment = QuantumHardwareExperiment(g_experiment, ydim, τs)
experiment_fidelity =
    QuantumSimulationExperiment(experimental_system, ψ̃_inits, gs_fidelity, τs)

experiment_full_state =
    QuantumSimulationExperiment(experimental_system, ψ̃_inits, gs_full_state, τs)

g_experiment(traj.a, times(traj), [traj.T]; print_result=true)

y_initial = experiment_fidelity(traj.a, timesteps(traj)).ys[end]
println("y_initial = ", y_initial)

J(y) = norm(y - y_goal, 1)

α = 1.0
β = 0.0
optimizer = BacktrackingLineSearch(J; α=α, β=β, verbose=true, global_min=false)

R_val = 0.15

R = (
    a=R_val,
    da=R_val,
    dda=R_val,
    Ũ⃗=R_val
)

prob = ILCProblem(
    traj,
    system,
    integrators,
    experiment_full_state,
    optimizer;
    g_ref=g_ref,
    max_iter=ilc_max_iter,
    state_name=:Ũ⃗,
    R=R
)

solve!(prob)

experiment_name = "pi_gate_ilc_R_val_$(R_val)"

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
println("y_final = ", y_final)

g_experiment(prob.Zs[end].a, times(prob.Zs[end]), [traj.T]; print_result=true)
