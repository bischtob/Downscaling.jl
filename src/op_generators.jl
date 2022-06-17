
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
    lifting
    downsampling
    nonlinear
    upsampling
    projection
    σ  
end

@functor UNetOperator

function UNetOperator(
    in_channels::Int,
    num_features::Int,
    num_modes::Tuple,
    num_subsample::Int,
    num_nonlinear::Int;
    lowpass_factor = 4,
    σ=gelu,
    transform = FourierTransform,
)
    # lifting
    lifting = []
    push!(lifting, Dense(in_channels, div(num_features, 2)))
    push!(lifting, Dense(div(num_features, 2), num_features))

    # downsampling
    downsampling = []
    for i in 1:num_subsample
        mode_contraction = div.(num_modes, Ref(lowpass_factor^i))
        push!(downsampling, OperatorKernel(num_features * 2^(i - 1) => num_features * 2^i, mode_contraction, transform))
    end

    # nonlinear bulk
    channel_expansion = (num_features * 2^num_subsample => num_features * 2^num_subsample)
    mode_contraction = div.(num_modes, Ref(lowpass_factor^num_subsample))
    nonlinear = Chain(
        [OperatorKernel(channel_expansion, mode_contraction, transform, σ) for _ in 1:num_nonlinear]...
    )

    # upsampling
    upsampling = []
    for i in reverse(1:num_subsample)
        if i == num_subsample
            channel_expansion = (num_features * 2^i => num_features * 2^(i-1))
        else
            #channel_expansion = (num_features * 2^(i+1) => num_features * 2^(i-1))
            channel_expansion = (num_features * 2^i => num_features * 2^(i-1))
        end
        mode_contraction = div.(num_modes, Ref(lowpass_factor^(i-1)))
        push!(upsampling, OperatorKernel(channel_expansion, mode_contraction, transform))
    end

    # projection
    projection = []
    push!(projection, Dense(2 * num_features, 3 * num_features))
    push!(projection, Dense(3 * num_features + div(num_features, 2), in_channels))

    return UNetOperator(
        lifting,
        downsampling,
        nonlinear,
        upsampling,
        projection,
        σ,
    )
end

function (op::UNetOperator)(x)
    # lifting
    l0 = op.σ.(op.lifting[1](x))
    l1 = op.σ.(op.lifting[2](l0))

    # downsampling and storing for bypass connections
    tmp = l1
    ds = []
    for (idx, layer) in enumerate(op.downsampling)
        tmp = op.σ.(layer(tmp))
        if idx < length(op.downsampling)
            push!(ds, tmp)
        end
    end

    # nonlinear layers without down- or upsampling
    tmp = op.nonlinear(tmp)

    # upsampling with adding bypass connections
    for (idx, layer) in enumerate(op.upsampling)
        if idx > 1
            bypass_idx = length(ds) - (idx - 1 - 1) 
            #tmp = vcat([tmp, ds[bypass_idx]]...)
        end
        tmp = op.σ.(layer(tmp))
    end

    # projection
    tmp = vcat([tmp, l1]...)
    lm1 = op.σ.(op.projection[1](tmp))
    lm1 = vcat([lm1, l0]...)
    lm0 = op.σ.(op.projection[2](lm1))

    return lm0
end
