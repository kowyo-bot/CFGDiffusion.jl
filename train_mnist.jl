using Pkg
Pkg.activate(@__DIR__)

using Flux
using Flux: @functor, gelu, softmax, batched_mul
using Zygote
using MLDatasets
using Statistics
using Random
using Printf
using ProgressMeter
using JLD2
using CairoMakie

# ==================== Utils ====================

function positional_embedding(time_steps::Int, dim::Int)
    half_dim = dim ÷ 2
    emb = log(10000.0f0) / (half_dim - 1)
    emb = exp.(-(0:half_dim-1) .* emb)
    pos = 0:time_steps-1
    emb = pos .* emb'
    emb = hcat(sin.(emb), cos.(emb))
    Float32.(emb)
end

# ==================== Conditional U-Net ====================

struct TimeEmbed
    mlp::Chain
end

function TimeEmbed(dim::Int)
    TimeEmbed(Chain(
        Dense(dim, dim * 4, gelu),
        Dense(dim * 4, dim * 4, gelu),
        Dense(dim * 4, dim)
    ))
end

function (te::TimeEmbed)(t_emb::AbstractMatrix)
    te.mlp(t_emb)
end

@functor TimeEmbed

struct ResBlock
    conv1::Conv
    conv2::Conv
    bn1::BatchNorm
    bn2::BatchNorm
    time_proj::Dense
    shortcut::Union{Conv,typeof(identity)}
end

function ResBlock(in_ch::Int, out_ch::Int, time_dim::Int)
    conv1 = Conv((3, 3), in_ch => out_ch, pad=1)
    conv2 = Conv((3, 3), out_ch => out_ch, pad=1)
    bn1 = BatchNorm(out_ch)
    bn2 = BatchNorm(out_ch)
    time_proj = Dense(time_dim, out_ch)
    shortcut = in_ch == out_ch ? identity : Conv((1, 1), in_ch => out_ch)
    
    ResBlock(conv1, conv2, bn1, bn2, time_proj, shortcut)
end

function (rb::ResBlock)(x, t_emb)
    h = rb.conv1(x)
    h = rb.bn1(h)
    h = gelu.(h)
    
    # Add time embedding
    t = rb.time_proj(t_emb)
    t = reshape(t, 1, 1, :, size(t, 2))
    h = h .+ t
    
    h = rb.conv2(h)
    h = rb.bn2(h)
    h = gelu.(h)
    
    h .+ rb.shortcut(x)
end

@functor ResBlock

struct SimpleUNet
    time_embed::Matrix{Float32}
    time_mlp::TimeEmbed
    
    # Encoder
    conv_in::Conv
    down1::ResBlock
    down2::ResBlock
    downsample::Conv
    
    # Bottleneck
    mid1::ResBlock
    mid2::ResBlock
    
    # Decoder
    upsample::ConvTranspose
    up2::ResBlock
    up1::ResBlock
    conv_out::Conv
    
    # Class embedding for CFG
    class_embed::Embedding
    num_classes::Int
end

function SimpleUNet(;
    in_ch::Int=1,
    out_ch::Int=1,
    base_ch::Int=64,
    time_dim::Int=128,
    num_classes::Int=10
)
    # Time embedding
    time_embed = positional_embedding(1000, time_dim)
    time_mlp = TimeEmbed(time_dim)
    
    # Class embedding (+1 for null class)
    class_embed = Embedding(num_classes + 1, time_dim)
    
    # Encoder
    conv_in = Conv((3, 3), in_ch => base_ch, pad=1)
    down1 = ResBlock(base_ch, base_ch, time_dim)
    down2 = ResBlock(base_ch, base_ch * 2, time_dim)
    downsample = Conv((4, 4), base_ch * 2 => base_ch * 2; stride=2, pad=1)
    
    # Bottleneck
    mid1 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
    mid2 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
    
    # Decoder
    upsample = ConvTranspose((4, 4), base_ch * 2 => base_ch * 2; stride=2, pad=1)
    up2 = ResBlock(base_ch * 4, base_ch, time_dim)
    up1 = ResBlock(base_ch * 2, base_ch, time_dim)
    conv_out = Conv((3, 3), base_ch => out_ch, pad=1)
    
    SimpleUNet(time_embed, time_mlp, conv_in, down1, down2, downsample,
               mid1, mid2, upsample, up2, up1, conv_out,
               class_embed, num_classes)
end

function (m::SimpleUNet)(x, t, c=nothing; guidance_scale::Float32=1.0f0)
    # x: (H, W, C, B)
    # t: (B,) time steps
    # c: (B,) class labels or nothing
    
    # Get time embedding
    t_emb_raw = m.time_embed[t, :]'
    t_emb = m.time_mlp(t_emb_raw)
    
    # Classifier-free guidance
    if c !== nothing
        # Conditional prediction
        # Map labels (0-9) to 2-11, null class (-1) to 1
        c_mapped = c .+ 2
        c_emb = m.class_embed(c_mapped)
        t_emb_cond = t_emb .+ c_emb'
        
        # Unconditional prediction (null class = 1)
        c_null = fill(1, size(c))
        c_emb_null = m.class_embed(c_null)
        t_emb_uncond = t_emb .+ c_emb_null'
        
        # Forward both
        noise_cond = forward_unet(m, x, t_emb_cond)
        noise_uncond = forward_unet(m, x, t_emb_uncond)
        
        # CFG: ε(x_t, c) = ε(x_t) + s(ε(x_t|c) - ε(x_t))
        return noise_uncond .+ guidance_scale .* (noise_cond .- noise_uncond)
    else
        return forward_unet(m, x, t_emb)
    end
end

function forward_unet(m::SimpleUNet, x, t_emb)
    # Encoder
    h = m.conv_in(x)
    h1 = m.down1(h, t_emb)
    h2 = m.down2(h1, t_emb)
    h = m.downsample(h2)
    
    # Bottleneck
    h = m.mid1(h, t_emb)
    h = m.mid2(h, t_emb)
    
    # Decoder
    h = m.upsample(h)
    h = cat(h, h2; dims=3)
    h = m.up2(h, t_emb)
    h = cat(h, h1; dims=3)
    h = m.up1(h, t_emb)
    
    m.conv_out(h)
end

@functor SimpleUNet

# ==================== DDPM ====================

struct DDPM
    T::Int
    beta::Vector{Float32}
    alpha::Vector{Float32}
    alpha_bar::Vector{Float32}
    sqrt_alpha_bar::Vector{Float32}
    sqrt_one_minus_alpha_bar::Vector{Float32}
    p_uncond::Float32  # Probability of unconditional training
end

function DDPM(T::Int=1000; p_uncond::Float32=0.1f0)
    beta = Float32.(range(1f-4, 2f-2, length=T))
    alpha = 1.0f0 .- beta
    alpha_bar = cumprod(alpha)
    
    DDPM(T, beta, alpha, alpha_bar,
         sqrt.(alpha_bar), sqrt.(1.0f0 .- alpha_bar), p_uncond)
end

function q_sample(ddpm::DDPM, x0::AbstractArray, t::AbstractVector{Int}, noise=nothing)
    if noise === nothing
        noise = randn!(similar(x0))
    end
    
    a = reshape(ddpm.sqrt_alpha_bar[t], 1, 1, 1, :)
    b = reshape(ddpm.sqrt_one_minus_alpha_bar[t], 1, 1, 1, :)
    
    a .* x0 .+ b .* noise
end

function p_losses(ddpm::DDPM, model, x0::AbstractArray, t::AbstractVector{Int}, labels::AbstractVector{Int})
    batch_size = size(x0, 4)
    noise = randn!(similar(x0))
    x_t = q_sample(ddpm, x0, t, noise)
    
    # Classifier-free guidance: randomly drop labels
    drop_mask = rand(Float32, batch_size) .< ddpm.p_uncond
    c = ifelse.(drop_mask, -1, labels)
    
    # Training with guidance_scale=1.0 (no guidance, just conditioning)
    noise_pred = model(x_t, t, c; guidance_scale=1.0f0)
    
    mean((noise_pred .- noise).^2)
end

function p_sample(ddpm::DDPM, model, x_t::AbstractArray, t::Int, c::Union{AbstractVector{Int},Nothing}; guidance_scale::Float32=1.0f0)
    beta_t = ddpm.beta[t]
    sqrt_one_minus_alpha_bar_t = ddpm.sqrt_one_minus_alpha_bar[t]
    sqrt_alpha_t = sqrt(ddpm.alpha[t])
    
    batch_size = size(x_t, 4)
    t_batch = fill(t, batch_size)
    
    noise_pred = model(x_t, t_batch, c; guidance_scale=guidance_scale)
    
    # Mean of p(x_{t-1} | x_t)
    model_mean = (x_t .- beta_t / sqrt_one_minus_alpha_bar_t .* noise_pred) / sqrt_alpha_t
    
    if t == 1
        model_mean
    else
        noise = randn!(similar(x_t))
        # Variance of p(x_{t-1} | x_t)
        var = beta_t
        model_mean .+ sqrt(var) .* noise
    end
end

function sample(ddpm::DDPM, model, shape::Tuple, c::Union{AbstractVector{Int},Nothing}; guidance_scale::Float32=1.0f0)
    x = randn(Float32, shape)
    
    @info "Sampling with guidance_scale=$guidance_scale..."
    prog = Progress(ddpm.T, desc="Sampling")
    for t in ddpm.T:-1:1
        x = p_sample(ddpm, model, x, t, c; guidance_scale=guidance_scale)
        next!(prog)
    end
    
    x
end

# ==================== Data ====================

function load_mnist(batch_size::Int=128)
    train_x, train_y = MNIST(:train)[:]
    test_x, test_y = MNIST(:test)[:]
    
    # Normalize to [-1, 1]
    train_x = Float32.(reshape(train_x, 28, 28, 1, :)) .* 2.0f0 .- 1.0f0
    test_x = Float32.(reshape(test_x, 28, 28, 1, :)) .* 2.0f0 .- 1.0f0
    
    (train_x, Int.(train_y)), (test_x, Int.(test_y))
end

function create_batches(x, y, batch_size::Int)
    n = size(x, 4)
    indices = shuffle(1:n)
    batches = []
    
    for i in 1:batch_size:n
        end_idx = min(i + batch_size - 1, n)
        batch_idx = indices[i:end_idx]
        push!(batches, (x[:, :, :, batch_idx], y[batch_idx]))
    end
    
    batches
end

# ==================== Training ====================

function train(; epochs=50, batch_size=128, lr=2e-4, device=cpu, save_dir="results")
    mkpath(save_dir)
    
    @info "Loading MNIST dataset..."
    (train_x, train_y), (test_x, test_y) = load_mnist(batch_size)
    @info "Train samples: $(size(train_x, 4)), Test samples: $(size(test_x, 4))"
    
    # Initialize model
    @info "Initializing model..."
    model = SimpleUNet(time_dim=128, num_classes=10) |> device
    ddpm = DDPM(1000, p_uncond=0.1f0)
    
    opt_state = Flux.setup(Adam(lr), model)
    
    losses = Float32[]
    best_loss = Inf32
    
    @info "Starting training for $epochs epochs..."
    for epoch in 1:epochs
        batches = create_batches(train_x, train_y, batch_size)
        epoch_losses = Float32[]
        
        prog = Progress(length(batches), desc="Epoch $epoch/$epochs")
        for (x_batch, y_batch) in batches
            x_batch = x_batch |> device
            y_batch = y_batch |> device
            
            t = rand(1:ddpm.T, size(x_batch, 4))
            
            loss, grads = Flux.withgradient(model) do m
                p_losses(ddpm, m, x_batch, t, y_batch)
            end
            
            Flux.update!(opt_state, model, grads[1])
            push!(epoch_losses, loss)
            next!(prog)
        end
        
        avg_loss = mean(epoch_losses)
        push!(losses, avg_loss)
        
        @info @sprintf("Epoch %3d/%d - Loss: %.6f", epoch, epochs, avg_loss)
        
        # Save checkpoint
        if avg_loss < best_loss
            best_loss = avg_loss
            model_cpu = model |> cpu
            @save joinpath(save_dir, "best_model.jld2") model_cpu
        end
        
        # Sample every 10 epochs
        if epoch % 10 == 0 || epoch == epochs
            @info "Generating samples..."
            visualize_samples(model |> cpu, ddpm, epoch, save_dir)
        end
    end
    
    # Save final
    model_cpu = model |> cpu
    @save joinpath(save_dir, "final_model.jld2") model_cpu
    
    @info "Training complete! Final loss: $(round(losses[end], digits=6))"
    
    # Plot loss curve
    plot_loss(losses, joinpath(save_dir, "loss_curve.png"))
    
    model, losses
end

# ==================== Visualization ====================

function visualize_samples(model, ddpm, epoch, save_dir)
    # Test different guidance scales
    guidance_scales = [1.0f0, 3.0f0, 7.5f0]
    
    for gs in guidance_scales
        all_samples = []
        for digit in 0:9
            labels = fill(digit, 8)
            x = sample(ddpm, model, (28, 28, 1, 8), labels; guidance_scale=gs)
            push!(all_samples, x)
        end
        
        # Create grid: 10 rows (digits) x 8 columns (samples)
        grid = make_grid(vcat(all_samples...), 10, 8)
        path = joinpath(save_dir, "epoch_$(lpad(epoch, 3, '0'))_gs$(gs).png")
        save_image(grid, path)
        @info "  → Saved: $path"
    end
end

function make_grid(images, rows, cols)
    h, w = size(images, 1), size(images, 2)
    grid = zeros(Float32, h * rows, w * cols)
    
    images = (images .+ 1.0f0) ./ 2.0f0  # [-1, 1] → [0, 1]
    images = clamp.(images, 0.0f0, 1.0f0)
    
    for i in 0:rows-1
        for j in 0:cols-1
            idx = i * cols + j + 1
            if idx <= size(images, 4)
                grid[i*h+1:(i+1)*h, j*w+1:(j+1)*w] = images[:, :, 1, idx]
            end
        end
    end
    
    grid
end

function save_image(grid, path)
    fig = Figure(size=(size(grid, 2), size(grid, 1)))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    image!(ax, grid; colormap=:grays, flip=true)
    save(path, fig)
end

function plot_loss(losses, path)
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss", title="Training Loss")
    lines!(ax, losses, linewidth=2)
    save(path, fig)
    @info "Saved loss curve: $path"
end

# ==================== Main ====================

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(42)
    
    # Detect device
    device = cpu
    @info "Using device: CPU"
    
    # Train with smaller epochs for testing
    train(epochs=50, batch_size=128, lr=2e-4, device=device, save_dir="results")
end
