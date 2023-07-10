using QuantumCollocation
import QuantumCollocation: lift
using NamedTrajectories
using LinearAlgebra
include(joinpath(@__DIR__, "qram_system.jl"))

# time parameters
duration = 170.0 # ns
T = 100
Δt = duration / T
Δt_max = 1.5 * Δt
Δt_min = 0.2 * Δt

warm_start = false


if warm_start
    data_path = joinpath(@__DIR__, "data_demo/unitary_levels_3_T_100_dt_2.054693946482075_dda_0.0001_a_0.02_max_iter_5000_00000.jld2")
    data = load_problem(data_path; return_data=true)
    init_traj = data["trajectory"]
    a_guess = init_traj.a
    Δt = init_traj.Δt[end]
end


# drive constraint: 20 MHz (linear units)
a_bound = 20e-3 # GHz

# pulse acceleration (used to control smoothness)
dda_bound = 1e-4

# maximum number of iterations
max_iter = 5000

# number of levels modeled for each qubit
levels = 3

# number of qubits
qubits = 2

α = [225.78, 100.33] * 1e-3 # GHz
# α = [0.1, 0.1] * 1e-3 # GHz

χ = -5.10982939 * 1e-3 # GHz

â_dag = create(levels)
â = annihilate(levels)

lift(op, i) = lift(op, i, qubits; l=levels)

# drift hamiltonian for ith qubit
H_q(i) = -α[i] / 2 * lift(â_dag, i)^2 * lift(â, i)^2

H_c_12 = χ * lift(â_dag, 1) * lift(â, 1) * lift(â_dag, 2) * lift(â, 2)

# drive hamiltonian for ith qubit, real part
H_d_real(i) = 1 / 2 * (lift(â_dag, i) + lift(â, i))

# drive hamiltonian for ith qubit, imaginary part
H_d_imag(i) = 1im / 2 * (lift(â_dag, i) - lift(â, i))

# total drift hamiltonian
H_drift = H_q(1) + H_q(2) + H_c_12
H_drift *= 2π

single_drive = false

if single_drive
    H_drives = Matrix{ComplexF64}.([H_d_real(1), H_d_imag(1)])
else
    H_drives = Matrix{ComplexF64}.([H_d_real(1), H_d_imag(1), H_d_real(2), H_d_imag(2)])
end
H_drives .*= 2π

# a_guess = collect(transpose(hcat(
#     randn(T),
#     randn(T),
#     fill(5e-3, T),
#     fill(5e-3, T),
# )))

# a_guess = zeros(4, T)

system = QuantumSystem(H_drift, H_drives)
new_system = QRAMSystem(qubits=[1, 2], drives=[1, 2])

@info "" new_system.G_drift ≈ system.G_drift
@info "" new_system.G_drives .≈ system.G_drives

# create goal unitary
Id = 1.0I(levels)
g = cavity_state(0, levels)
e = cavity_state(1, levels)
eg = e * g'
ge = g * e'
ee = e * e'
gg = g * g'
U_goal = (ee + gg) ⊗ (eg + ge)
U_init = (ee + gg) ⊗ (ee + gg)

ψ_inits = [
    g ⊗ g,
    g ⊗ e,
    g ⊗ (e + g) / √2 ,
    g ⊗ (e + im * g) / √2,
    e ⊗ g,
    e ⊗ e,
    e ⊗ (e + g) / √2,
    e ⊗ (e + im * g) / √2,
    (-e ⊗ e + im * e ⊗ g + g ⊗ g + im * g ⊗ e) / 2,
]

ψ̃_inits = ket_to_iso.(ψ_inits)
ψ_goals = [U_goal * ψ_init for ψ_init in ψ_inits]
ψ̃_goals = ket_to_iso.(ψ_goals)

prob = UnitarySmoothPulseProblem(
    system,
    U_goal,
    T,
    Δt;
    U_init=U_init,
    Δt_max=Δt_max,
    Δt_min=Δt_min,
    a_bound=a_bound,
    dda_bound=dda_bound,
    a_guess=warm_start ? a_guess : nothing,
    max_iter=max_iter,
)


# prob = QuantumStateSmoothPulseProblem(
#     system,
#     ψ_inits,
#     ψ_goals,
#     T,
#     Δt;
#     Δt_max=Δt_max,
#     Δt_min=Δt_min,
#     a_bound=a_bound,
#     dda_bound=dda_bound,
#     max_iter=max_iter,
#     a_guess=warm_start ? a_guess : nothing,
# )

plot_dir = joinpath(@__DIR__, "plots_demo")
save_dir = joinpath(@__DIR__, "data_demo")

experiment_name = "unitary_$(single_drive ? "single_drive" : "two_drives")_levels_$(levels)_T_$(T)_dt_$(Δt)_dda_$(dda_bound)_a_$(a_bound)_max_iter_$(max_iter)"

plot_path = generate_file_path("png", experiment_name, plot_dir)
save_path = generate_file_path("jld2", experiment_name, save_dir)

# plot the initial guess for the wavefunction and controls
plot(plot_path, prob.trajectory, [:Ũ⃗, :a]; ignored_labels=[:Ũ⃗])
# plot(plot_path, prob.trajectory, [:ψ̃1, :ψ̃2, :a])

solve!(prob)

# plot the final solution for the wavefunction and controls
plot(plot_path, prob.trajectory, [:Ũ⃗, :a]; ignored_labels=[:Ũ⃗])
# plot(plot_path, prob.trajectory, [:ψ̃1, :ψ̃2, :a])

# test pulse
A = prob.trajectory.a
Δt = prob.trajectory.Δt

Ũ⃗_final = unitary_rollout(A, Δt, system; integrator=exp)[:, end]

final_unitary_fidelity = unitary_fidelity(iso_vec_to_operator(Ũ⃗_final), U_goal)
println("Final unitary fidelity: $final_unitary_fidelity")


# ψ̃_finals_old = [
#     rollout(ψ̃_init, A, Δt, old_system; integrator=exp)[:, end]
#         for ψ̃_init in ψ̃_inits
# ]

# final_fidelities_old = [
#     iso_fidelity(ψ̃_final, ψ̃_goal)
#         for (ψ̃_final, ψ̃_goal) in zip(ψ̃_finals_old, ψ̃_goals)
# ]
# println("Final fidelities: $final_fidelities_old")



ψ̃_finals = [
    rollout(ψ̃_init, A, Δt, system; integrator=exp)[:, end]
        for ψ̃_init in ψ̃_inits
]

final_fidelities = [
    iso_fidelity(ψ̃_final, ψ̃_goal)
        for (ψ̃_final, ψ̃_goal) in zip(ψ̃_finals, ψ̃_goals)
]
println("Final fidelities: $final_fidelities")

duration = times(prob.trajectory)[end]

info = Dict(
    "final_fidelities" => final_fidelities,
    "duration" => duration,
)

# save the solution
save_problem(save_path, prob, info)
