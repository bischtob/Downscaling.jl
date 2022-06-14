
using Flux
using Functors

struct UNetGenerator
    initial
    downblocks
    resblocks
    upblocks
    final
end

@functor UNetGenerator

function UNetGenerator(
    in_channels::Int,
    num_features::Int=64,
    num_residual::Int=9,
    device=gpu,
)
    initial_layer = Chain(
        Conv((7, 7), in_channels => num_features; stride=1, pad=3),
        InstanceNorm(num_features),
        x -> relu.(x)
    )

    downsampling_blocks = [
        ConvBlock(3, num_features, num_features * 2, true, true, device; stride=2, pad=1),
        ConvBlock(3, num_features * 2, num_features * 4, true, true, device; stride=2, pad=1),
    ]

    resnet_blocks = Chain([ResidualBlock(num_features * 4) for _ in range(1, length=num_residual)]...)

    upsampling_blocks = [
        ConvBlock(3, num_features * 4, num_features * 2, true, false, device; stride=2, pad=SamePad()),
        ConvBlock(3, num_features * 2, num_features, true, false, device; stride=2, pad=SamePad()),
    ]

    final_layer = Chain(
        Conv((7, 7), num_features => in_channels; stride=1, pad=3)
    )

    return UNetGenerator(
        initial_layer,
        downsampling_blocks,
        resnet_blocks,
        upsampling_blocks,
        final_layer
    ) |> device
end

function (net::UNetGenerator)(x)
    input = net.initial(x)
    for layer in net.downblocks
        input = layer(input)
    end
    input = net.resblocks(input)
    for layer in net.upblocks
        input = layer(input)
    end
    return tanh.(net.final(input))
end

"""
    ConvBlock
"""
struct ConvBlock
    conv
end

@functor ConvBlock

function ConvBlock(
    kernel_size::Int,
    in_channels::Int,
    out_channels::Int,
    with_activation::Bool=true,
    down::Bool=true,
    device=gpu;
    kwargs...
)
    return ConvBlock(
        Chain(
            if down
                Conv((kernel_size, kernel_size), in_channels => out_channels; kwargs...)
            else
                ConvTranspose((kernel_size, kernel_size), in_channels => out_channels; kwargs...)
            end,
            InstanceNorm(out_channels),
            if with_activation
                x -> relu.(x)
            else
                identity
            end)
    ) |> device
end

function (net::ConvBlock)(x)
    return net.conv(x)
end


"""
    ResidualBlock
"""
struct ResidualBlock
    block
end

@functor ResidualBlock

function ResidualBlock(
    in_channels::Int,
    device=gpu
)
    return ResidualBlock(
        Chain(
            ConvBlock(3, in_channels, in_channels, true, true, device; pad=1),
            ConvBlock(3, in_channels, in_channels, false, true, device; pad=1)
        )
    ) |> device
end

function (net::ResidualBlock)(x)
    return x + net.block(x)
end
