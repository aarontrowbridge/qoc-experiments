using QuantumCollocation
import QuantumCollocation: lift
using NamedTrajectories
using LinearAlgebra

include(joinpath(@__DIR__, "qram_system.jl"))

qubits = [1, 2]
levels = [2, 2]
drives = [1, 2]
gate = :CX

system = QRAMSystem(; qubits=qubits, levels=levels, drives=drives)

U_init, U_goal = qram_subspace_unitary(levels, gate, [1,2])
display(U_goal)

# time parameters
duration = 200.0 # ns
T = 100
Δt = duration / T
Δt_max = 1.5 * Δt
Δt_min = 0.5 * Δt

# drive constraint: 20 MHz (linear units)
a_bound = 20 * 1e-3 # GHz

# pulse acceleration (used to control smoothness)
dda_bound = 1e-3

# maximum number of iterations
max_iter = 500

# warm start
warm_start = false

if warm_start
    data_path = joinpath(@__DIR__, "data/limited_drives_T_100_dt_1.0_dda_0.002_a_0.02_max_iter_500_00001.jld2")
    data = load_problem(data_path; return_data=true)
    init_traj = data["trajectory"]
    init_drives = init_traj.a
    init_Δt = init_traj.Δt[end]
end

prob = UnitarySmoothPulseProblem(
    system,
    U_goal,
    T,
    warm_start ? init_Δt : Δt;
    U_init=U_init,
    Δt_max=Δt_max,
    Δt_min=Δt_min,
    a_bound=a_bound,
    dda_bound=dda_bound,
    max_iter=max_iter,
    a_guess=warm_start ? init_drives : nothing,
)

save_dir = joinpath(@__DIR__, "data")
plot_dir = joinpath(@__DIR__, "plots")

experiment_name = "gate_$(gate)_levels_$(join(levels, "_"))_drives_$(join(drives, "_"))_T_$(T)_dt_$(Δt)_dda_$(dda_bound)_a_$(a_bound)_max_iter_$(max_iter)"

save_path = generate_file_path("jld2", experiment_name, save_dir)
plot_path = generate_file_path("png", experiment_name, plot_dir)

# plot the initial guess for the wavefunction and controls
plot(plot_path, prob.trajectory, [:a])

solve!(prob)
A = prob.trajectory.a
Δt = prob.trajectory.Δt

Ũ⃗_final = unitary_rollout(A, Δt, system; integrator=exp)[:, end]
final_fidelity = unitary_fidelity(Ũ⃗_final, prob.trajectory.goal.Ũ⃗)
println("Final fidelity: $final_fidelity")


# plot the final solution for the wavefunction and controls
plot(plot_path, prob.trajectory, [:Ũ⃗, :a])



duration = times(prob.trajectory)[end]

info = Dict(
    "final_fidelity" => final_fidelity,
    "duration" => duration,
)



# save the solution
save_problem(save_path, prob, info)
