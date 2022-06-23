
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
    UNetOperator
"""
struct UNetOperator
    network
end

@functor UNetOperator

function UNetOperator(n_channel::Int, n_codim::Int, n_modes::Int; trafo=FourierTransform, σ=gelu)
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
                            OperatorKernel(n_codim=>2n_codim, (n_modes÷4, n_modes÷4), trafo, σ),
                            SkipConnection(
                                Chain(
                                    OperatorKernel(2n_codim=>4n_codim, (n_modes÷16, n_modes÷16), trafo, σ),
                                    SkipConnection(
                                        Chain(
                                            OperatorKernel(4n_codim=>8n_codim, (n_modes÷64, n_modes÷64), trafo, σ),
                                            OperatorKernel(8n_codim=>8n_codim, (n_modes÷64, n_modes÷64), trafo, σ),
                                            OperatorKernel(8n_codim=>4n_codim, (n_modes÷16, n_modes÷16), trafo, σ),
                                        ),
                                        vcat,
                                    ),
                                    OperatorKernel(8n_codim=>2n_codim, (n_modes÷4, n_modes÷4), trafo, σ),
                                ),
                                vcat,
                            ),
                            OperatorKernel(4n_codim=>n_codim, (n_modes, n_modes), trafo, σ),
                        ),
                        vcat,
                    ),
                    Dense(2n_codim, 3n_codim, σ),
                ),
                vcat,
            ),
            Dense(3n_codim + n_codim÷2, n_channel, σ),
        )

    return UNetOperator(network)
end

function (op::UNetOperator)(x)
    return op.network(x)
end
