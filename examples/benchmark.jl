using Flux
using BenchmarkTools
import Downscaling: UNetOperator

# benchmark UNetOperator
img_size = 128
batch_size = 1
n_channel = 1
n_codim = 32
n_modes = 64
x = rand(Float32, n_channel, img_size, img_size, batch_size) |> gpu
op = UNetOperator(
    n_channel,
    n_codim,
    n_modes,
) |> gpu
op(x) |> size

# gradient check
loss = () -> sum(op(x))
ps = Flux.params(op)
gs = gradient(loss, ps)
