
using Flux
using Functors
using NeuralOperators
using Statistics

# """
#     UNetOperatorGenerator
# """
# struct UNetOperatorGenerator
#     unet
# end

# @functor UNetOperatorGenerator

# function UNetOperatorGenerator(
#     in_channels::Int,
#     num_features::Int,
#     num_modes::Tuple,
#     num_subsample::Int,
#     num_nonlinear::Int;
#     σ=gelu,
#     kwargs...
# )
#     return UNetOperatorGenerator(
#         UNetOperator(
#             in_channels,
#             num_features,
#             num_modes,
#             num_subsample,
#             num_nonlinear,
#             σ=σ,
#             kwargs...
#          )
#     )
# end

# function (op::UNetOperatorGenerator)(x)
#     return tanh.(op.unet(x))
# end

# """
#     UNetOperatorDiscriminator
# """
# struct UNetOperatorDiscriminator
#     unet
# end

# @functor UNetOperatorDiscriminator

# function UNetOperatorDiscriminator(
#     in_channels::Int,
#     num_features::Int,
#     num_modes::Tuple,
#     num_subsample::Int,
#     num_nonlinear::Int;
#     σ=gelu,
#     kwargs...
# )
#     return UNetOperatorDiscriminator(
#         UNetOperator(
#             in_channels,
#             num_features,
#             num_modes,
#             num_subsample,
#             num_nonlinear,
#             σ=σ,
#             kwargs...
#         )
#     )
# end

# function (op::UNetOperatorDiscriminator)(x)
#     return sigmoid.(mean(op.unet(x)))
# end

"""
    UNetOperator2D
"""
struct UNetOperator2D
    network
end

@functor UNetOperator2D

function UNetOperator2D(n_channel::Int, n_codim::Int, n_modes::Int; trafo=FourierTransform, σ=gelu)
    @assert n_codim ÷ 2 > 0
    @assert n_modes ÷ 64 > 0

    network = 
        Chain(
            Dense(n_channel, n_codim ÷ 2, σ),
            SkipConnection(
                Chain(
                    Dense(n_codim ÷ 2, n_codim, σ),
                    SkipConnection(
                        Chain(
                            OperatorBlock2D(n_codim=>2n_codim, (n_modes÷4, n_modes÷4), trafo, σ),
                            SkipConnection(
                                Chain(
                                    OperatorBlock2D(2n_codim=>4n_codim, (n_modes÷16, n_modes÷16), trafo, σ),
                                    SkipConnection(
                                        Chain(
                                            OperatorBlock2D(4n_codim=>8n_codim, (n_modes÷64, n_modes÷64), trafo, σ),
                                            OperatorBlock2D(8n_codim=>8n_codim, (n_modes÷64, n_modes÷64), trafo, σ),
                                            OperatorBlock2D(8n_codim=>4n_codim, (n_modes÷16, n_modes÷16), trafo, σ),
                                        ),
                                        vcat,
                                    ),
                                    OperatorBlock2D(8n_codim=>2n_codim, (n_modes÷4, n_modes÷4), trafo, σ),
                                ),
                                vcat,
                            ),
                            OperatorBlock2D(4n_codim=>n_codim, (n_modes, n_modes), trafo, σ),
                        ),
                        vcat,
                    ),
                    Dense(2n_codim, 3n_codim, σ),
                ),
                vcat,
            ),
            Dense(3n_codim + n_codim÷2, n_channel),
        )

    return UNetOperator2D(network)
end

# function UNetOperator2D(n_channel::Int, n_codim::Int, n_modes::Int; trafo=FourierTransform, σ=gelu)
#     @assert n_codim ÷ 2 > 0
#     @assert n_modes ÷ 64 > 0

#     network =
#         Chain(
#             Dense(n_channel, n_codim ÷ 2, σ),
#             SkipConnection(
#                 Chain(
#                     Dense(n_codim ÷ 2, n_codim, σ),
#                     SkipConnection(
#                         Chain(
#                             OperatorBlock2D(n_codim => 2n_codim, (n_modes ÷ 4, n_modes ÷ 4), trafo, σ),
#                             SkipConnection(
#                                 Chain(
#                                     OperatorBlock2D(2n_codim => 4n_codim, (n_modes ÷ 16, n_modes ÷ 16), trafo, σ),
#                                     SkipConnection(
#                                         Chain(
#                                             OperatorBlock2D(4n_codim => 8n_codim, (n_modes ÷ 64, n_modes ÷ 64), trafo, σ),
#                                             OperatorBlock2D(8n_codim => 8n_codim, (n_modes ÷ 64, n_modes ÷ 64), trafo, σ),
#                                             OperatorBlock2D(8n_codim => 4n_codim, (n_modes ÷ 16, n_modes ÷ 16), trafo, σ),
#                                         ),
#                                         vcat,
#                                     ),
#                                     OperatorBlock2D(8n_codim => 2n_codim, (n_modes ÷ 4, n_modes ÷ 4), trafo, σ),
#                                 ),
#                                 vcat,
#                             ),
#                             OperatorBlock2D(4n_codim => n_codim, (n_modes, n_modes), trafo, σ),
#                         ),
#                         vcat,
#                     ),
#                     Dense(2n_codim, 3n_codim, σ),
#                 ),
#                 vcat,
#             ),
#             Dense(3n_codim + n_codim ÷ 2, n_channel),
#         )

#     return UNetOperator2D(network)
# end

function (op::UNetOperator2D)(x)
    return op.network(x)
end


"""
    OperatorBlock2D
"""
struct OperatorBlock2D
    network
end

@functor OperatorBlock2D

function OperatorBlock2D(channels, modes, trafo=FourierTransform, σ=gelu)
    _, out_channels = channels

    network = Chain(
        OperatorKernel(channels, modes, trafo, identity),
        x -> permutedims(x, (3, 2, 1, 4)),
        InstanceNorm(out_channels),
        x -> permutedims(x, (3, 2, 1, 4)),
        x -> σ.(x),
    )

    return OperatorBlock2D(network)
end

function (op::OperatorBlock2D)(x)
    return op.network(x)
end
