using Flux
using BenchmarkTools
import Downscaling: UNetOperator2D

function benchmark()
    # benchmark UNetOperator
    img_size = 64
    batch_size = 20
    n_channel = 1
    n_codim = 32
    n_modes = 64
    x = rand(Float32, n_channel, img_size, img_size, batch_size) |> gpu
    op = UNetOperator2D(
        n_channel,
        n_codim,
        n_modes,
    ) |> gpu
    #op(x) |> size |> println

    # gradient check
    loss = () -> sum(op(x))
    ps = Flux.params(op)
    gs = gradient(loss, ps)

    nothing
end

@benchmark benchmark()
