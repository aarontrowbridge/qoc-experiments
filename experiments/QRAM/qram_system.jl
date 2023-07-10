using QuantumCollocation
import QuantumCollocation: lift
using NamedTrajectories
using LinearAlgebra

function QuantumCollocation.lift(
    op::AbstractMatrix{<:Number},
    i::Int,
    levels::Vector{Int}
)
    @assert size(op, 1) == size(op, 2) == levels[i] "Operator must be square and match dimension of qubit i"

    Is = [collect(1.0I(l)) for l ∈ levels]
    Is[i] = op
    return kron(Is...)
end

find(qubits, q) = findfirst(qubits .== q)

function QRAMSystem(;
    qubits=[1,2,3,4],
    drives=[1,2,3,4],
    levels=fill(3, length(qubits)),
    α=[225.78, 100.33, 189.32, 172.15] * 1e-3, # GHz (linear units)
    χ=Symmetric([
        0 -5.10982939 -0.18457118 -0.50235316;
        0       0     -0.94914758 -1.07618574;
        0       0           0     -0.44607489;
        0       0           0           0
    ]) * 1e-3, # GHz (linear units)
)
    @assert length(levels) == length(qubits)
    @assert unique(qubits) == qubits
    @assert unique(drives) == drives
    @assert all(drive ∈ qubits for drive ∈ drives)

    â_dag(i) = create(levels[find(qubits, i)])
    â(i) = annihilate(levels[find(qubits, i)])

    H_q = sum(
        -α[i] / 2 * lift(â_dag(i)^2 * â(i)^2, find(qubits, i), levels)
            for i ∈ qubits
    )

    H_c = sum(
        χ[i, j] *
        lift(â_dag(i) * â(i), find(qubits, i), levels) *
        lift(â_dag(j) * â(j), find(qubits, j), levels)
            for i ∈ qubits, j ∈ qubits if j > i
    )

    H_drift = 2π * (H_q + H_c)

    H_d_real(i) = 1 / 2 * lift(â_dag(i) + â(i), find(qubits, i), levels)

    H_d_imag(i) = 1.0im / 2 * lift(â_dag(i) - â(i), find(qubits, i), levels)

    H_drives::Vector{Matrix{ComplexF64}} =
        vcat([[H_d_real(i), H_d_imag(i)] for i ∈ drives]...)

    H_drives .*= 2π

    @assert all(H == H' for H ∈ H_drives)

    return QuantumSystem(H_drift, H_drives)
end

function qram_subspace_unitary(
    levels::Vector{Int},
    gate::AbstractMatrix{<:Number},
    qubit::Union{Int, Vector{Int}}
)
    if qubit isa Int
        @assert size(gate, 1) == size(gate, 2) == 2 "Gate must be a single-qubit gate"
    else
        @assert all(qubit .== qubit[1]:qubit[end]) "Qubits must be consecutive"
        @assert size(gate, 1) == size(gate, 2) == 2^length(qubit) "Gate must be a multi-qubit gate"
    end

    U_init = []
    U_goal = []
    Ids = []
    for (i, level) ∈ enumerate(levels)
        gᵢ = cavity_state(0, level)
        eᵢ = cavity_state(1, level)
        Idᵢ =  gᵢ * gᵢ' + eᵢ * eᵢ'
        push!(Ids, Idᵢ)
    end

        if i == qubit
            push!(U_init, Idᵢ)
            Uᵢ = zeros(ComplexF64, level, level)
            Uᵢ[1:2, 1:2] = gate
            push!(U_goal, Uᵢ)
        else
            push!(U_init, Idᵢ)
            push!(U_goal, Idᵢ)
        end
    end
    U_init = kron(Ids...)
    U_goal = kron(U_goal...)
    return U_init, U_goal
end

function qram_subspace_unitary(
    levels::Vector{Int},
    gate::Symbol,
    qubit::Int
)
    U = GATES[gate]
    return qram_subspace_unitary(levels, U, qubit)
end
