
using Flux
using Functors
using NeuralOperators

"""
    UNetDiscriminator
"""
struct UNetDiscriminator
    unet
    knet
end
#         kx = self.knet(grid)
        kx = kx.view(batch_size,-1, 1)
        x = x.view(batch_size,-1, 1)
        x = torch.einsum('bik,bik->bk', kx, x)/(res1*res2)


@functor UNetDiscriminator

function (op::UNetDiscriminator)(x)
    return op.knet(op.unet(x))
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
        Dense(in_channels, num_features),
        x -> σ.(x),
        Dense(num_features, num_features),
    )

    downsampling = [
        OperatorBlock(num_features * 2^(i - 1), num_features * 2^i, div.(num_modes, Ref(2^i)), σ=σ, kwargs...) for i = 1:num_subsample
    ]

    nonlinear = Chain(
        [OperatorBlock(num_features * 2^num_subsample, div.(num_modes, Ref(2^num_subsample)), σ=σ, kwargs...) for _ in range(1, length=num_nonlinear)]...
    )

    upsampling = [
        OperatorBlock(num_features * 2^i, num_features * 2^(i - 1), div.(num_modes, Ref(2^i)), σ=σ, kwargs...) for i in reverse(1:num_subsample)
    ]

    projection = Chain(
        Dense(num_features, num_features),
        x -> σ.(x),
        Dense(num_features, in_channels),
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
    input = op.lifting(x)

    ds_layers = []
    for layer in op.downsampling
        input = layer(input)
        push!(ds_layers, input)
    end

    input = net.nonlinear(input)

    for layer in net.upsampling
        input = Flux.stack([layer(input), pop!(ds_layers)], dims=1)
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
    return ConvBlock(
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

function (net::OperatorConvBlock)(x)
    return net.block(x)
end
