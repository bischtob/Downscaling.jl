
using Flux
using Functors
using NeuralOperators
using Statistics


"""
    UNetOperatorGenerator
"""
struct UNetOperatorGenerator
    unet
end

@functor UNetOperatorGenerator

function UNetOperatorGenerator(
    in_channels::Int,
    num_features::Int,
    num_modes::Tuple,
    num_subsample::Int,
    num_nonlinear::Int;
    σ=gelu,
    kwargs...
)
    return UNetOperatorGenerator(
        UNetOperator(
            in_channels,
            num_features,
            num_modes,
            num_subsample,
            num_nonlinear,
            σ=σ,
            kwargs...
         )
    )
end

function (op::UNetOperatorGenerator)(x)
    return tanh.(op.unet(x))
end

"""
    UNetOperatorDiscriminator
"""
struct UNetOperatorDiscriminator
    unet
end

@functor UNetOperatorDiscriminator

function UNetOperatorDiscriminator(
    in_channels::Int,
    num_features::Int,
    num_modes::Tuple,
    num_subsample::Int,
    num_nonlinear::Int;
    σ=gelu,
    kwargs...
)
    return UNetOperatorDiscriminator(
        UNetOperator(
            in_channels,
            num_features,
            num_modes,
            num_subsample,
            num_nonlinear,
            σ=σ,
            kwargs...
        )
    )
end

function (op::UNetOperatorDiscriminator)(x)
    return sigmoid.(mean(op.unet(x)))
end


"""
    UNetOperator
"""
struct UNetOperator
    lifting
    downsampling
    nonlinear
    upsampling
    projection
end

@functor UNetOperator

function UNetOperator(
    in_channels::Int,
    num_features::Int,
    num_modes::Tuple,
    num_subsample::Int,
    num_nonlinear::Int;
    σ=gelu,
    kwargs...
)
    lifting = Chain(
        Dense(in_channels, div(num_features, 2)),
        x -> σ.(x),
        Dense(div(num_features, 2), num_features),
    )

    downsampling = [
        OperatorBlock(num_features * 2^(i - 1), num_features * 2^i, div.(num_modes, Ref(2^i)), σ=σ, kwargs...) for i = 1:num_subsample
    ]

    nonlinear = Chain(
        [OperatorBlock(num_features * 2^num_subsample, num_features * 2^num_subsample, div.(num_modes, Ref(2^num_subsample)), σ=σ, kwargs...) for _ in range(1, length=num_nonlinear)]...
    )

    upsampling = []
    for i in reverse(1:num_subsample)
        if i == num_subsample
            push!(upsampling, OperatorBlock(num_features * 2^i, num_features * 2^(i - 1), div.(num_modes, Ref(2^(i - 1))), σ=σ, kwargs...))
        else
            push!(upsampling, OperatorBlock(num_features * 2^(i + 1), num_features * 2^(i - 1), div.(num_modes, Ref(2^(i - 1))), σ=σ, kwargs...))
        end
    end

    projection = Chain(
        Dense(num_features, 2 * num_features),
        x -> σ.(x),
        Dense(2 * num_features, in_channels),
    )

    return UNetOperator(
        lifting,
        downsampling,
        nonlinear,
        upsampling,
        projection,
    )
end

function (op::UNetOperator)(x)
    input = permutedims(x, (3, 2, 1, 4))
    input = op.lifting(input)

    # downsampling and storing for bypass connections
    ds_layers = []
    for (idx, layer) in enumerate(op.downsampling)
        input = layer(input)
        if idx < length(op.downsampling)
            push!(ds_layers, input)
        end
    end

    # nonlinear layers without down- or upsampling
    input = op.nonlinear(input)

    # upsampling with adding bypass connections
    for (idx, layer) in enumerate(op.upsampling)
        if idx < length(op.upsampling)
            input = vcat([layer(input), pop!(ds_layers)]...)
        else
            input = layer(input)
        end
    end

    input = op.projection(input)

    return permutedims(input, (3, 2, 1, 4))
end


"""
    OperatorBlock
"""
struct OperatorBlock
    block
end

@functor OperatorBlock

function OperatorBlock(
    in_channels::Int,
    out_channels::Int,
    modes::Tuple;
    transform=FourierTransform,
    σ=gelu,
    permuted=false,
    normed=false
)
    return OperatorBlock(
        Chain(
            OperatorKernel(in_channels => out_channels, modes, transform, σ, permuted=permuted),
            if normed
                InstanceNorm(out_channels)
            else
                identity
            end
        )
    )
end

function (net::OperatorBlock)(x)
    return net.block(x)
end
