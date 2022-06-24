module Downscaling

using Flux
using Flux.Optimise: update!
using Flux.Losses: mse, mae
using Functors: @functor
using Statistics
using Zygote

include("generators.jl")
include("discriminators.jl")
include("op_generators.jl")

# nns
export ConvBlock
export ResidualBlock
export PatchBlock
export PatchDiscriminator
export UNetGenerator

# nos
export OperatorBlock2D
export UNetOperator2D
export UNetOperatorGenerator
export UNetOperatorDiscriminator

end
