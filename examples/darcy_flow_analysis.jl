using BSON
using CUDA
using Flux
using FluxTraining
using MAT
using MLUtils

using Downscaling
using NeuralOperators
using UnicodePlots: heatmap

function get_dataloader(path; split_ratio=0.5, batchsize=4, sampling_rate=2)
    # TODO! Make this HDF5 based stuff, wtf. single file path!
    file = MAT.matopen(path)
    X, Y = read(file, "coeff"), read(file, "sol")
    X = permutedims(X[:, :, :, :], (4, 2, 3, 1))[:, 1:sampling_rate:end, 1:sampling_rate:end, :]
    Y = permutedims(Y[:, :, :, :], (4, 2, 3, 1))[:, 1:sampling_rate:end, 1:sampling_rate:end, :]
    close(file)

    data_training, data_validation = MLUtils.splitobs((X, Y), at=split_ratio)
    loader_training = Flux.DataLoader(data_training, batchsize=batchsize, shuffle=true)
    loader_validation = Flux.DataLoader(data_validation, batchsize=batchsize, shuffle=true)

    return (training=loader_training, validation=loader_validation,)
end

function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

# check out the model
_, data_val = get_dataloader("../data/darcy/data_file1.mat"; split_ratio=0.8, batchsize=1)
op = get_model()

X, Y = first(data_val)
Y_pred = op(X)

X = X[1, :, :, 1]
Y = Y[1, :, :, 1]
Y_pred = Y_pred[1, :, :, 1]
