module CFGDiffusion

using Flux
using Zygote
using LinearAlgebra
using Statistics
using Random
using Printf

export UNet, ConditionalUNet
export DDPM, DDPMLoss
export p_losses, sample
export p_sample, p_sample_cfg

# ==================== Time Embedding ====================

struct SinusoidalPositionEmbedding
    embedding::Matrix{Float32}
end

function SinusoidalPositionEmbedding(time_steps::Int, dim::Int)
    half_dim = dim ÷ 2
    emb = log(10000.0f0) / (half_dim - 1)
    emb = exp.(-(0:half_dim-1) .* emb)
    pos = 0:time_steps-1
    emb = pos .* emb'
    emb = hcat(sin.(emb), cos.(emb))
    SinusoidalPositionEmbedding(Float32.(emb))
end

function (spe::SinusoidalPositionEmbedding)(t::AbstractVector{Int})
    spe.embedding[t, :]'  # (dim, batch)
end

# ==================== Conditional U-Net for MNIST ====================

struct TimeEmbeddingLayer
    linear1::Dense
    linear2::Dense
    activation
end

function TimeEmbeddingLayer(in_dim::Int, out_dim::Int)
    TimeEmbeddingLayer(
        Dense(in_dim, out_dim, gelu),
        Dense(out_dim, out_dim),
        gelu
    )
end

function (tel::TimeEmbeddingLayer)(t_emb)
    h = tel.linear1(t_emb)
    h = tel.activation(h)
    tel.linear2(h)
end

struct ResidualBlock
    conv1::Conv
    conv2::Conv
    norm1::BatchNorm
    norm2::BatchNorm
    time_mlp::TimeEmbeddingLayer
    shortcut::Union{Conv,typeof(identity)}
end

function ResidualBlock(in_ch::Int, out_ch::Int, time_emb_dim::Int)
    conv1 = Conv((3, 3), in_ch => out_ch, pad=1)
    conv2 = Conv((3, 3), out_ch => out_ch, pad=1)
    norm1 = BatchNorm(out_ch)
    norm2 = BatchNorm(out_ch)
    time_mlp = TimeEmbeddingLayer(time_emb_dim, out_ch)
    shortcut = in_ch == out_ch ? identity : Conv((1, 1), in_ch => out_ch)
    
    ResidualBlock(conv1, conv2, norm1, norm2, time_mlp, shortcut)
end

function (rb::ResidualBlock)(x, t_emb)
    h = rb.conv1(x)
    h = rb.norm1(h)
    h = gelu.(h)
    
    # Add time embedding
    t_out = rb.time_mlp(t_emb)
    t_out = reshape(t_out, 1, 1, :, size(t_out, 2))
    h = h .+ t_out
    
    h = rb.conv2(h)
    h = rb.norm2(h)
    h = gelu.(h)
    
    h .+ rb.shortcut(x)
end

struct AttentionBlock
    qkv::Conv
    proj::Conv
    norm::GroupNorm
    num_heads::Int
end

function AttentionBlock(channels::Int, num_heads::Int=4)
    qkv = Conv((1, 1), channels => channels * 3)
    proj = Conv((1, 1), channels => channels)
    norm = GroupNorm(32, channels)
    AttentionBlock(qkv, proj, norm, num_heads)
end

function (ab::AttentionBlock)(x)
    b, h, w, c, n = size(x, 1), size(x, 2), size(x, 3), size(x, 4), size(x, 5)
    
    x_norm = ab.norm(x)
    qkv = ab.qkv(x_norm)
    
    # Reshape for multi-head attention
    qkv = reshape(qkv, h*w, 3, ab.num_heads, c ÷ ab.num_heads, n)
    q, k, v = qkv[:, 1, :, :, :], qkv[:, 2, :, :, :], qkv[:, 3, :, :, :]
    
    # Scaled dot-product attention
    scale = sqrt(Float32(size(k, 3)))
    attn = softmax(batched_mul(q, permutedims(k, (1, 3, 2, 4))) ./ scale, dims=2)
    out = batched_mul(attn, v)
    
    out = reshape(out, h, w, c, n)
    out = ab.proj(out)
    x .+ out
end

struct DownBlock
    resblock::ResidualBlock
    attn::Union{AttentionBlock,Nothing}
    downsample::Union{Conv,Nothing}
end

function DownBlock(in_ch::Int, out_ch::Int, time_emb_dim::Int; downsample::Bool=true, use_attn::Bool=false)
    resblock = ResidualBlock(in_ch, out_ch, time_emb_dim)
    attn = use_attn ? AttentionBlock(out_ch) : nothing
    ds = downsample ? Conv((4, 4), out_ch => out_ch; stride=2, pad=1) : nothing
    DownBlock(resblock, attn, ds)
end

function (db::DownBlock)(x, t_emb)
    h = db.resblock(x, t_emb)
    if db.attn !== nothing
        h = db.attn(h)
    end
    if db.downsample !== nothing
        h = db.downsample(h)
    end
    h
end

struct UpBlock
    upsample::Union{ConvTranspose,Nothing}
    resblock::ResidualBlock
    attn::Union{AttentionBlock,Nothing}
end

function UpBlock(in_ch::Int, out_ch::Int, time_emb_dim::Int; upsample::Bool=true, use_attn::Bool=false)
    us = upsample ? ConvTranspose((4, 4), in_ch => in_ch; stride=2, pad=1) : nothing
    resblock = ResidualBlock(in_ch, out_ch, time_emb_dim)
    attn = use_attn ? AttentionBlock(out_ch) : nothing
    UpBlock(us, resblock, attn)
end

function (ub::UpBlock)(x, skip, t_emb)
    if ub.upsample !== nothing
        x = ub.upsample(x)
    end
    # Concatenate skip connection
    x = cat(x, skip; dims=3)
    h = ub.resblock(x, t_emb)
    if ub.attn !== nothing
        h = ub.attn(h)
    end
    h
end

struct UNet
    time_embed::SinusoidalPositionEmbedding
    time_mlp::Chain
    
    input_conv::Conv
    down_blocks::Vector{DownBlock}
    mid_block1::ResidualBlock
    mid_attn::AttentionBlock
    mid_block2::ResidualBlock
    up_blocks::Vector{UpBlock}
    output_conv::Conv
    
    num_classes::Int
    class_embed::Embedding  # 类别嵌入
end

function UNet(;
    in_channels::Int=1,
    out_channels::Int=1,
    base_channels::Int=64,
    channel_mult::Vector{Int}=[1, 2, 4],
    num_res_blocks::Int=2,
    time_emb_dim::Int=256,
    num_classes::Int=10,
    dropout::Float32=0.1f0
)
    # Time embedding
    time_embed = SinusoidalPositionEmbedding(1000, time_emb_dim)
    time_mlp = Chain(
        Dense(time_emb_dim, time_emb_dim * 4, gelu),
        Dense(time_emb_dim * 4, time_emb_dim * 4, gelu),
        Dense(time_emb_dim * 4, time_emb_dim)
    )
    
    # Class embedding for conditional generation
    class_embed = Embedding(num_classes + 1, time_emb_dim)  # +1 for null class
    
    # Input
    input_conv = Conv((3, 3), in_channels => base_channels, pad=1)
    
    # Down blocks
    down_blocks = DownBlock[]
    ch = base_channels
    for (i, mult) in enumerate(channel_mult)
        out_ch = base_channels * mult
        for _ in 1:num_res_blocks
            push!(down_blocks, DownBlock(ch, out_ch, time_emb_dim; 
                downsample=false, use_attn=(i >= 2)))
            ch = out_ch
        end
        if i < length(channel_mult)
            push!(down_blocks, DownBlock(ch, ch, time_emb_dim; 
                downsample=true, use_attn=false))
        end
    end
    
    # Middle
    mid_block1 = ResidualBlock(ch, ch, time_emb_dim)
    mid_attn = AttentionBlock(ch)
    mid_block2 = ResidualBlock(ch, ch, time_emb_dim)
    
    # Up blocks
    up_blocks = UpBlock[]
    for (i, mult) in enumerate(reverse(channel_mult))
        out_ch = base_channels * mult
        for j in 1:(num_res_blocks + 1)
            push!(up_blocks, UpBlock(ch, out_ch, time_emb_dim;
                upsample=(j == 1 && i > 1), use_attn=(i <= 2)))
            ch = out_ch
        end
    end
    
    # Output
    output_conv = Conv((3, 3), base_channels => out_channels, pad=1)
    
    UNet(time_embed, time_mlp, input_conv, down_blocks, 
         mid_block1, mid_attn, mid_block2, up_blocks, output_conv,
         num_classes, class_embed)
end

function (unet::UNet)(x, t, c=nothing; guidance_scale::Float32=1.0f0)
    # x: (H, W, C, B)
    # t: (B,) time steps
    # c: (B,) class labels or nothing
    
    # Time embedding
    t_emb_raw = unet.time_embed(t)
    t_emb = unet.time_mlp(t_emb_raw')
    
    # Classifier-free guidance
    if c !== nothing
        # Conditional prediction
        # Map labels (0-9) to 2-11, null class (-1) to 1
        c_mapped = c .+ 2
        c_emb = unet.class_embed(c_mapped)
        t_emb_cond = t_emb .+ c_emb'
        
        # Unconditional prediction (null class = index 1)
        c_null = fill(1, size(c))
        c_emb_null = unet.class_embed(c_null)
        t_emb_uncond = t_emb .+ c_emb_null'
        
        # Both forward passes
        noise_cond = unet_forward(unet, x, t_emb_cond)
        noise_uncond = unet_forward(unet, x, t_emb_uncond)
        
        # CFG formula: ε(x_t, c) = ε(x_t) + s(ε(x_t|c) - ε(x_t))
        return noise_uncond .+ guidance_scale .* (noise_cond .- noise_uncond)
    else
        return unet_forward(unet, x, t_emb)
    end
end

function unet_forward(unet::UNet, x, t_emb)
    # Input
    h = unet.input_conv(x)
    
    # Down path with skip connections
    skips = []
    for block in unet.down_blocks
        h = block(h, t_emb)
        push!(skips, h)
    end
    
    # Middle
    h = unet.mid_block1(h, t_emb)
    h = unet.mid_attn(h)
    h = unet.mid_block2(h, t_emb)
    
    # Up path
    for block in unet.up_blocks
        skip = pop!(skips)
        h = block(h, skip, t_emb)
    end
    
    unet.output_conv(h)
end

# ==================== DDPM ====================

struct DDPM
    num_timesteps::Int
    betas::Vector{Float32}
    alphas::Vector{Float32}
    alphas_cumprod::Vector{Float32}
    sqrt_alphas_cumprod::Vector{Float32}
    sqrt_one_minus_alphas_cumprod::Vector{Float32}
    sqrt_recip_alphas::Vector{Float32}
    posterior_variance::Vector{Float32}
    p_uncond::Float32  # Probability of unconditional training
end

function DDPM(num_timesteps::Int=1000; beta_start::Float32=1f-4, beta_end::Float32=2f-2, p_uncond::Float32=0.1f0)
    betas = Float32.(range(beta_start, beta_end, length=num_timesteps))
    alphas = 1.0f0 .- betas
    alphas_cumprod = cumprod(alphas)
    alphas_cumprod_prev = vcat([1.0f0], alphas_cumprod[1:end-1])
    
    sqrt_alphas_cumprod = sqrt.(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = sqrt.(1.0f0 .- alphas_cumprod)
    sqrt_recip_alphas = sqrt.(1.0f0 ./ alphas)
    
    posterior_variance = betas .* (1.0f0 .- alphas_cumprod_prev) ./ (1.0f0 .- alphas_cumprod)
    
    DDPM(num_timesteps, betas, alphas, alphas_cumprod,
         sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
         sqrt_recip_alphas, posterior_variance, p_uncond)
end

function q_sample(ddpm::DDPM, x0::AbstractArray, t::AbstractVector{Int}, noise=nothing)
    if noise === nothing
        noise = randn!(similar(x0))
    end
    
    sqrt_alpha_t = ddpm.sqrt_alphas_cumprod[t]
    sqrt_one_minus_alpha_t = ddpm.sqrt_one_minus_alphas_cumprod[t]
    
    sqrt_alpha_t = reshape(sqrt_alpha_t, 1, 1, 1, :)
    sqrt_one_minus_alpha_t = reshape(sqrt_one_minus_alpha_t, 1, 1, 1, :)
    
    sqrt_alpha_t .* x0 .+ sqrt_one_minus_alpha_t .* noise
end

function p_losses(ddpm::DDPM, model, x0::AbstractArray, t::AbstractVector{Int}, labels::AbstractVector{Int})
    batch_size = size(x0, 4)
    noise = randn!(similar(x0))
    x_t = q_sample(ddpm, x0, t, noise)
    
    # Classifier-free guidance training: randomly drop labels
    drop_mask = rand(Float32, batch_size) .< ddpm.p_uncond
    c = ifelse.(drop_mask, -1, labels)
    
    # For training, we use guidance_scale=1.0 (no guidance, just conditioning)
    noise_pred = model(x_t, t, c; guidance_scale=1.0f0)
    
    mean((noise_pred .- noise).^2)
end

function p_sample(ddpm::DDPM, model, x_t::AbstractArray, t::Int, c::Union{AbstractVector{Int},Nothing}; guidance_scale::Float32=1.0f0)
    betas_t = ddpm.betas[t]
    sqrt_one_minus_alpha_t = ddpm.sqrt_one_minus_alphas_cumprod[t]
    sqrt_recip_alpha_t = ddpm.sqrt_recip_alphas[t]
    
    batch_size = size(x_t, 4)
    t_batch = fill(t, batch_size)
    
    noise_pred = model(x_t, t_batch, c; guidance_scale=guidance_scale)
    
    model_mean = sqrt_recip_alpha_t .* (x_t .- betas_t ./ sqrt_one_minus_alpha_t .* noise_pred)
    
    if t == 1
        model_mean
    else
        noise = randn!(similar(x_t))
        posterior_variance_t = ddpm.posterior_variance[t]
        model_mean .+ sqrt(posterior_variance_t) .* noise
    end
end

function sample(ddpm::DDPM, model, shape::Tuple, labels::Union{AbstractVector{Int},Nothing}; guidance_scale::Float32=1.0f0)
    x = randn(Float32, shape)
    
    @info "Sampling..."
    for t in ddpm.num_timesteps:-1:1
        x = p_sample(ddpm, model, x, t, labels; guidance_scale=guidance_scale)
    end
    
    x
end

function sample_cfg(ddpm::DDPM, model, shape::Tuple, labels::AbstractVector{Int}; guidance_scale::Float32=7.5f0)
    sample(ddpm, model, shape, labels; guidance_scale=guidance_scale)
end

end # module
