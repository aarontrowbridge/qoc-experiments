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
