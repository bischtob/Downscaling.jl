using BSON
using CUDA
using Flux
using FluxTraining
using MAT
using MLUtils

using Downscaling: UNetOperator2D


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

function loss(y_pred, y_actual)
    spatial_dims = 2:(ndims(y_pred)-1)

    l2_dist_norms = sqrt.(sum(abs2, y_pred - y_actual, dims=spatial_dims))
    l2_y_actual_norms = sqrt.(sum(abs2, y_actual, dims=spatial_dims))

    return sum(l2_dist_norms ./ l2_y_actual_norms)
end

function fit!(learner, nepochs::Int)
    for _ in 1:nepochs
        FluxTraining.epoch!(learner, FluxTraining.TrainingPhase(), learner.data.training)
        FluxTraining.epoch!(learner, FluxTraining.ValidationPhase(), learner.data.validation)
    end
end

function train(; cuda=true, lr=1.0f-3, λ=1.0f-3, nepochs=400)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    data = get_dataloader("../data/darcy/data_file1.mat"; split_ratio=0.8, batchsize=5)
    model = UNetOperator2D(1, 32, 64)
    loss_func = loss
    optimizer = Flux.Optimiser(
        Flux.ADAM(lr), 
        WeightDecay(λ),
        ExpDecay(1, 0.7, 100, 1e-6)
    )

    learner = FluxTraining.Learner(
        model,
        data,
        optimizer,
        loss_func,
        FluxTraining.ToDevice(device, device),
        FluxTraining.Checkpointer(joinpath(@__DIR__, "../model/"))
    )

    fit!(learner, nepochs)

    return learner
end

function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

train()
