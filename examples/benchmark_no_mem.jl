using CUDA
using Flux
using BenchmarkTools
using NeuralOperators
using Downscaling: UNetOperator2D


function benchmark()
    # benchmark UNetOperator
    img_size = 211
    batch_size = 4
    n_channel = 1
    n_codim = 32
    n_modes = 64
    x = rand(Float32, n_channel, img_size, img_size, batch_size) |> gpu
    op = UNetOperator2D(
        n_channel,
        n_codim,
        n_modes,
    ) |> gpu

    # gradient check
    loss = () -> sum(op(x))
    ps = Flux.params(op)
    gs = gradient(loss, ps)

    nothing
end

GC.gc(); CUDA.reclaim(); CUDA.memory_status()
CUDA.@time benchmark()
CUDA.memory_status()

