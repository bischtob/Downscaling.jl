using Base.Iterators: partition
using BSON: @save, @load
using Dates
using Statistics: mean
using Images
using Flux
using Flux: params, update!
using Flux.Data: DataLoader
using NeuralOperators
using Random
using Downscaling: PatchDiscriminator, OperatorUNetGenerator

# Parameters
FT = Float32
exp_name = "moist2d"
input_path = "../data/"
output_path = "./output/"
eval_freq = 350
checkpoint_freq = eval_freq
device = gpu
num_examples = 700
num_epochs = 100
batch_size = 1
img_size = 256
input_channels = 1
dis_lr = FT(0.0002)
gen_lr = FT(0.0002)
λ = FT(10.0)
λid = FT(0.5)
color_format = Gray

# Define models
num_features = 8
σ = relu
generator_A = OperatorUNetGenerator(input_channels) |> device
generator_B = OperatorUNetGenerator(input_channels) |> device
discriminator_A = PatchDiscriminator(input_channels) |> device # Discriminator For Domain A
discriminator_B = PatchDiscriminator(input_channels) |> device # Discriminator For Domain B
networks = (generator_A, generator_B, discriminator_A, discriminator_B)

function generator_loss(a, b)
    a_fake = generator_B(b) # Fake image generated in domain A
    b_fake = generator_A(a) # Fake image generated in domain B

    b_fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    a_fake_prob = discriminator_A(a_fake) # Probability that generated image in domain A is real

    gen_A_loss = mean((a_fake_prob .- 1) .^ 2)
    rec_A_loss = mean(abs.(b - generator_A(a_fake))) # Cycle-consistency loss for domain B
    idt_A_loss = mean(abs.(generator_A(b) .- b)) # Identity loss for domain B
    gen_B_loss = mean((b_fake_prob .- 1) .^ 2)
    rec_B_loss = mean(abs.(a - generator_B(b_fake))) # Cycle-consistency loss for domain A
    idt_B_loss = mean(abs.(generator_B(a) .- a)) # Identity loss for domain A

    return gen_A_loss + gen_B_loss + λ * (rec_A_loss + rec_B_loss + λid * (idt_A_loss + idt_B_loss))
end

function discriminator_loss(a, b)
    a_fake = generator_B(b) # Fake image generated in domain A
    b_fake = generator_A(a) # Fake image generated in domain B

    a_fake_prob = discriminator_A(a_fake) # Probability that generated image in domain A is real
    a_real_prob = discriminator_A(a) # Probability that an original image in domain A is real
    b_fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    b_real_prob = discriminator_B(b) # Probability that an original image in domain B is real

    real_A_loss = mean((a_real_prob .- 1) .^ 2)
    fake_A_loss = mean((a_fake_prob .- 0) .^ 2)
    real_B_loss = mean((b_real_prob .- 1) .^ 2)
    fake_B_loss = mean((b_fake_prob .- 0) .^ 2)

    return real_A_loss + fake_A_loss + real_B_loss + fake_B_loss
end

function train_step(opt_gen, opt_dis, a, b)
    # Optimize Discriminators
    ps = params(params(discriminator_A)..., params(discriminator_B)...)
    gs = gradient(() -> discriminator_loss(a, b), ps)
    update!(opt_dis, ps, gs)

    # Optimize Generators
    ps = params(params(generator_A)..., params(generator_B)...)
    gs = gradient(() -> generator_loss(a, b), ps)
    update!(opt_gen, ps, gs)

    # Forward propagate to collect the losses for keeping track of training progress
    g_loss = generator_loss(a, b)
    d_loss = discriminator_loss(a, b)

    return g_loss, d_loss
end

function training()
    # Load data
    dataA = load_dataset(input_path * exp_name * "/trainA/", img_size, FT, color_format)[:, :, :, 1:num_examples] |> device
    dataB = load_dataset(input_path * exp_name * "/trainB/", img_size, FT, color_format)[:, :, :, 1:num_examples] |> device
    data = DataLoader((dataA, dataB), batchsize=batch_size, shuffle=true)

    # Define Optimizers
    opt_gen = ADAM(gen_lr, (0.5, 0.999))
    opt_dis = ADAM(dis_lr, (0.5, 0.999))

    # Training loop
    total_iters = 0
    @info "Training begins..."
    for epoch in 1:num_epochs
        epoch_start = Dates.now()
        @info "Epoch: $epoch -------------------------------------------------------------------"
        for (batch_idx, (a, b)) in enumerate(data)
            a, b = normalize(a), normalize(b)
            g_loss, d_loss = train_step(opt_gen, opt_dis, a, b)
            total_iters += batch_size

            if total_iters % eval_freq == 0
                @info "Total iteration: $total_iters - Generator loss: $g_loss, Discriminator loss: $d_loss"
            end

            if total_iters % checkpoint_freq == 0
                @info "Total iteration: $total_iters - Checkpointing model."
                file_last = output_path * exp_name * "/checkpoint_latest.bson"
                networks_cpu = networks |> cpu
                @save file_last networks_cpu

                prefix_path = output_path * exp_name * "/training/" * "image_$(total_iters)"
                save_model_samples(prefix_path, a, b)
            end
        end
        @info "Epoch duration: $(Dates.canonicalize(Dates.now() - epoch_start))"
    end
end

# function testing()
#     # Load data
#     dataA = load_dataset(input_path * exp_name * "/testA/", img_size, FT)[:, :, :, 1:num_examples] |> device
#     dataB = load_dataset(input_path * exp_name * "/testB/", img_size, FT)[:, :, :, 1:num_examples] |> device
#     data = DataLoader((dataA, dataB), batchsize=1, shuffle=false)

#     # Load generators
#     genA, genB, _, _ = @load output_path * exp_name * "checkpint_latest.bson"
#     Flux.loadmodel!(generator_A, genA)
#     Flux.loadmodel!(generator_B, genB)

#     # Testing loop
#     @info "Testing begins..."
#     for (sample_idx, (a, b)) in enumerate(data)
#         a, b = normalize(a), normalize(b)
#         prefix_path = output_path * exp_name * "/testing/" * "image_$(sample_idx)"
#         save_model_samples(path, a, b)
#     end
#     @info "Testing complete."
# end

# Utils
function normalize(x)
    return @. 2 * x - 1
end

function unnormalize(x)
    return @. (x + 1) / 2
end

function load_image(path, img_size=256, FT=Float32, color_format=RGB)
    # channelview to convert to 3 channels
    img = path |> load .|> color_format |> channelview .|> FT
    if length(size(img)) == 3
        img = permutedims(img, (3, 2, 1))
    end

    return imresize(img, (img_size, img_size))
end

function save_image(path, img, color_format=RGB)
    if size(img)[3] == 3
        img = permutedims(img, (3, 2, 1))
    elseif size(img)[3] == 1
        img = img[:, :, 1]
    end
    img = colorview(color_format, img)
    save(path, img)
end

function load_dataset(path, img_size=256, FT=Float32, color_format=RGB)
    loader = path -> load_image(path, img_size, FT, color_format)
    imgs = map(loader, path .* readdir(path))

    return Flux.stack(imgs, dims=4)
end

function save_model_samples(prefix_path, a, b)
    fake_B = generator_A(a)
    fake_A = generator_B(b)
    rec_B = generator_A(fake_A)
    rec_A = generator_B(fake_B)

    # Save the images
    imgs = [a, fake_A, rec_A, b, fake_B, rec_B] |> cpu
    imgs = map(unnormalize, imgs)
    suffices = ["real_A", "fake_A", "rec_A", "real_B", "fake_B", "rec_B"]
    for (img_batch, suffix) in zip(imgs, suffices)
        for img_idx in 1:size(img_batch)[end]
            path = prefix_path * "_$(suffix)_$(img_idx).png"
            save_image(path, img_batch[:, :, :, img_idx], color_format)
        end
    end
end

training()
