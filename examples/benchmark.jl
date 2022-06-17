using Flux
using BenchmarkTools
import Downscaling: UNetOperator

# benchmark UNetOperator
img_size = 256
batch_size = 1
in_channels = 1
num_features = 32
num_modes = (128, 128)
num_subsample = 3
num_nonlinear = 1
x = rand(Float32, in_channels, img_size, img_size, batch_size) |> gpu
op = UNetOperator(
    in_channels,
    num_features,
    num_modes,
    num_subsample,
    num_nonlinear,
) |> gpu
op(x);

# gradient check
loss = () -> sum(op(x))
ps = Flux.params(op)
gs = gradient(loss, ps)
