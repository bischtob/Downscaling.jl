using Downscaling
using Flux
using Test
using Random

float_types = [Float16, Float32]

@testset "Downscaling.jl" begin
    for FT in float_types
        # PatchBlock
        in_channels = 3
        out_channels = 7
        img_size = 97
        batch_size = 13
        stride = 2
        patch_block = PatchBlock(in_channels, out_channels, stride=stride)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test patch_block(x) |> size == (48, 48, 7, 13)

        # PatchDiscriminator
        in_channels = 3
        batch_size = 13
        patch_dsicriminator = PatchDiscriminator(in_channels)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test patch_dsicriminator(x) |> size == (10, 10, 1, 13)

        # ConvBlock
        kernel_size = 3
        in_channels = 3
        out_channels = 7
        with_activation = true
        down = true
        batch_size = 13
        img_size = 97
        conv_block = ConvBlock(kernel_size, in_channels, out_channels, with_activation, down)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test conv_block(x) |> size == (95, 95, 7, 13)

        kernel_size = 3
        in_channels = 3
        out_channels = 7
        with_activation = true
        down = false
        batch_size = 13
        img_size = 97
        conv_block = ConvBlock(kernel_size, in_channels, out_channels, with_activation, down)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test conv_block(x) |> size == (99, 99, 7, 13)

        # ResidualBlock
        in_channels = 3
        batch_size = 13
        img_size = 97
        residual_block = ResidualBlock(in_channels)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test residual_block(x) |> size == (97, 97, 3, 13)

        # UNetGenerator
        img_size = 128
        batch_size = 5
        in_channels = 1
        num_features = 64
        num_residual = 9
        unet = UNetGenerator(in_channels, num_features, num_residual)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test unet(x) |> size == (img_size, img_size, in_channels, batch_size)

        # CycleGAN
        img_size = 128
        batch_size = 5
        in_channels = 3
        dAB = PatchDiscriminator(in_channels)
        dBA = PatchDiscriminator(in_channels)
        gAB = UNetGenerator(in_channels)
        gBA = UNetGenerator(in_channels)
        cyclegan = CycleGAN(gAB, gBA, dAB, dBA)
        opt_dscr = ADAM(0.00005, (0.5, 0.99))
        opt_gen = ADAM(0.00005, (0.5, 0.99))
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        y = randn(FT, (img_size, img_size, in_channels, batch_size))
        loss = update_cyclegan(opt_gen, opt_dscr, cyclegan, x, y)
    end
end