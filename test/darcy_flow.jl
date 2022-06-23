using MAT
using MLUtils
using Flux

function get_dataloader(path; split_ratio=0.5, batchsize=4)
    # TODO! Make this HDF5 based stuff, wtf. single file path!
    file = matopen(path)
    X, Y = read(file, "coeff"), read(file, "sol")
    X = permutedims(X[:, :, :, :], (4, 2, 3, 1))
    Y = permutedims(Y[:, :, :, :], (4, 2, 3, 1))
    close(file)

    data_train, data_test = splitobs((X, Y), at=split_ratio)
    loader_train = DataLoader(data_train, batchsize=batchsize, shuffle=true)
    loader_test = DataLoader(data_test, batchsize=batchsize, shuffle=true)

    return (train=loader_train, test=loader_test,)
end

train, test = get_dataloader("../data/darcy/data_file1.mat"; split_ratio=0.8)

struct SuperResPhase <: FluxTraining.AbstractValidationPhase end

FluxTraining.phasedataiter(::SuperResPhase) = :super_res

function FluxTraining.step!(learner, phase::SuperResPhase, batch)
    xs, ys = batch
    FluxTraining.runstep(learner, phase, (xs=xs, ys=ys)) do _, state
        state.ŷs = learner.model(state.xs)
        state.loss = learner.lossfn(state.ŷs, state.ys)
    end
end

function fit!(learner, nepochs::Int, (loader_train, loader_validate, loader_super_res))
    for i in 1:nepochs
        epoch!(learner, TrainingPhase(), loader_train)
        epoch!(learner, ValidationPhase(), loader_validate)
        epoch!(learner, SuperResPhase(), loader_super_res)
    end
end

function fit!(learner, nepochs::Int)
    fit!(learner, nepochs, (learner.data.training, learner.data.validation, learner.data.super_res))
end

function train(; cuda=true, η₀=1.0f-3, λ=1.0f-4, epochs=50)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    model = MarkovNeuralOperator(ch=(1, 64, 64, 64, 64, 64, 1), modes=(24, 24), σ=gelu)
    data = get_dataloader()
    optimiser = Flux.Optimiser(WeightDecay(λ), Flux.ADAM(η₀))
    loss_func = l₂loss

    learner = Learner(
        model, data, optimiser, loss_func,
        ToDevice(device, device),
        Checkpointer(joinpath(@__DIR__, "../model/"))
    )

    fit!(learner, epochs)

    return learner
end

function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end
