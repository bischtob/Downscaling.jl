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
    
        # OperatorBlock
        in_channels = 3
        out_channels = 7
        modes = (4, 4)
        batch_size = 13
        img_size = 97
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        block = OperatorBlock(in_channels, out_channels, modes, permuted=true)
        @test block(x) |> size == (97, 97, 7, 13)
        conv_block = OperatorBlock(in_channels, out_channels, modes, permuted=true, normed=true)
        @test block(x) |> size == (97, 97, 7, 13)
    
        # UNetOperator
        in_channels = 3
        num_features = 12
        modes = (4, 4)
        num_subsample = 4
        num_nonlinear = 4
        batch_size = 13
        img_size = 97
        op = UNetOperator(in_channels, num_features, modes, num_subsample, num_nonlinear)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test op(x) |> size == (97, 97, 3, 13)
    end
end