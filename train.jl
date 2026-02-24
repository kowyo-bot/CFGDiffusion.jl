using Pkg
Pkg.activate(@__DIR__)

using Flux
using Zygote
using MLDatasets
using Statistics
using Random
using Printf
using ProgressMeter
using CairoMakie

include("src/CFGDiffusion.jl")
using .CFGDiffusion

# ==================== Data Loading ====================

function load_mnist(batch_size::Int=128)
    train_x, train_y = MNIST(:train)[:]
    test_x, test_y = MNIST(:test)[:]
    
    # Normalize to [-1, 1]
    train_x = Float32.(reshape(train_x, 28, 28, 1, :)) .* 2.0f0 .- 1.0f0
    test_x = Float32.(reshape(test_x, 28, 28, 1, :)) .* 2.0f0 .- 1.0f0
    
    train_y = Int.(train_y)
    test_y = Int.(test_y)
    
    (train_x, train_y), (test_x, test_y)
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

function train(; epochs=100, batch_size=128, lr=2e-4, save_dir="checkpoints")
    mkpath(save_dir)
    
    @info "Loading MNIST dataset..."
    (train_x, train_y), (test_x, test_y) = load_mnist(batch_size)
    @info "Train samples: $(size(train_x, 4)), Test samples: $(size(test_x, 4))"
    
    # Initialize model
    @info "Initializing model..."
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mult=[1, 2, 4],
        num_res_blocks=2,
        time_emb_dim=256,
        num_classes=10,
        dropout=0.1f0
    ) |> gpu
    
    ddpm = DDPM(1000, p_uncond=0.1f0)
    
    opt_state = Flux.setup(Adam(lr), model)
    
    losses = Float32[]
    best_loss = Inf32
    
    @info "Starting training..."
    for epoch in 1:epochs
        batches = create_batches(train_x, train_y, batch_size)
        epoch_losses = Float32[]
        
        progress = Progress(length(batches), desc="Epoch $epoch/$epochs")
        for (x_batch, y_batch) in batches
            x_batch = x_batch |> gpu
            y_batch = y_batch |> gpu
            
            # Sample random timesteps
            t = rand(1:ddpm.num_timesteps, size(x_batch, 4))
            
            # Compute loss
            loss, grads = Flux.withgradient(model) do m
                p_losses(ddpm, m, x_batch, t, y_batch)
            end
            
            Flux.update!(opt_state, model, grads[1])
            push!(epoch_losses, loss)
            next!(progress)
        end
        
        avg_loss = mean(epoch_losses)
        push!(losses, avg_loss)
        
        @info @sprintf("Epoch %3d/%d - Loss: %.6f", epoch, epochs, avg_loss)
        
        # Save best model
        if avg_loss < best_loss
            best_loss = avg_loss
            model_cpu = model |> cpu
            jldsave(joinpath(save_dir, "best_model.jld2"); model_state=Flux.state(model_cpu))
            @info "  → Saved best model (loss: $(round(best_loss, digits=6)))"
        end
        
        # Sample every 10 epochs
        if epoch % 10 == 0 || epoch == 1
            @info "Generating samples..."
            sample_and_visualize(model, ddpm, epoch, save_dir)
        end
    end
    
    # Save final model
    model_cpu = model |> cpu
    jldsave(joinpath(save_dir, "final_model.jld2"); model_state=Flux.state(model_cpu))
    
    @info "Training complete! Final loss: $(round(losses[end], digits=6))"
    
    model, losses
end

function sample_and_visualize(model, ddpm, epoch, save_dir)
    model = model |> cpu
    
    # Sample each digit with different guidance scales
    guidance_scales = [1.0f0, 3.0f0, 7.5f0, 10.0f0]
    
    for gs in guidance_scales
        samples = []
        for digit in 0:9
            labels = fill(digit, 8)  # 8 samples per digit
            x = sample(ddpm, model, (28, 28, 1, 8), labels; guidance_scale=gs)
            push!(samples, x)
        end
        
        # Save visualization
        save_grid(vcat(samples...), 10, 8, joinpath(save_dir, "epoch_$(lpad(epoch, 3, '0'))_gs$(gs).png"))
    end
    
    model = model |> gpu
end

function save_grid(images, nrows, ncols, path)
    # images: (H, W, C, N)
    h, w = size(images, 1), size(images, 2)
    
    # Denormalize from [-1, 1] to [0, 1]
    images = (images .+ 1.0f0) ./ 2.0f0
    images = clamp.(images, 0.0f0, 1.0f0)
    
    # Create grid
    grid = zeros(Float32, h * nrows, w * ncols)
    for i in 0:nrows-1
        for j in 0:ncols-1
            idx = i * ncols + j + 1
            if idx <= size(images, 4)
                grid[i*h+1:(i+1)*h, j*w+1:(j+1)*w] = images[:, :, 1, idx]
            end
        end
    end
    
    # Save using CairoMakie
    fig = Figure(size=(ncols*32, nrows*32))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    image!(ax, grid; colormap=:grays)
    save(path, fig)
    @info "  → Saved: $path"
end

# ==================== Main ====================

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(42)
    train(epochs=100, batch_size=128, lr=2e-4)
end
