using Random
using MDPs
import MDPs: state_space, action_space, action_meaning, reset!, step!, in_absorbing_state

export GridWorldContinuous

mutable struct GridWorldContinuous{T} <: AbstractMDP{Vector{T}, Int}
    gw::GridWorld
    include_tile_type::Bool

    const 𝕊::VectorSpace{T}

    state::Vector{T}
    action::Int
    reward::Float64

    tile_type_to_onehot::Dict{Char, Vector{T}}

    function GridWorldContinuous{T}(gw::GridWorld, include_tile_type::Bool=false) where T<:AbstractFloat
        rows, cols = size(gw.grid)
        tile_types = gw.enter_rewards |> keys
        ntiletypes = length(tile_types)
        tile_type_to_onehot = Dict{Char, Vector{T}}()
        for (i, tile_type) in enumerate(tile_types)
            tile_type_to_onehot[tile_type] = zeros(T, ntiletypes)
            tile_type_to_onehot[tile_type][i] = 1
        end
        if include_tile_type
            𝕊 = VectorSpace{T}(T[1/rows, 1/cols, zeros(ntiletypes)...], T[1, 1, ones(ntiletypes)...])
            s = zeros(T, 2+ntiletypes)
        else
            𝕊 = VectorSpace{T}(T[1/rows, 1/cols], T[1, 1])
            s = zeros(T, 2)
        end
        new{T}(gw, include_tile_type, 𝕊, s, 0, 0.0, tile_type_to_onehot)
    end
end

function fractional_coordinates(gwc::GridWorldContinuous, s::Int)::Tuple{Float64, Float64}
    rcindex(gwc.gw, s) ./ size(gwc.gw.grid)
end
function factored_representation(gw::GridWorldContinuous{T}, s::Int)::Vector{T} where T
    if gw.include_tile_type
        return convert(Vector{T}, vcat(fractional_coordinates(gw, s)..., gw.tile_type_to_onehot[gw.gw.grid[s]]))
    else
        return T[fractional_coordinates(gw, s)...]
    end
end

@inline state_space(gwc::GridWorldContinuous) = gwc.𝕊
@inline action_space(gwc::GridWorldContinuous) = action_space(gwc.gw)
@inline action_meaning(gwc::GridWorldContinuous, a::Int) = action_meaning(gwc.gw, a)

function reset!(gwc::GridWorldContinuous{T}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where T
    reset!(gwc.gw; rng=rng)
    s = state(gwc.gw)
    gwc.state .= factored_representation(gwc, s)
    gwc.action = action(gwc.gw)
    gwc.reward = reward(gwc.gw)
    nothing
end

function step!(gwc::GridWorldContinuous, a::Int; rng=Random.AbstractRNG)::Nothing
    @assert a ∈ action_space(gwc)
    gwc.action = a
    if in_absorbing_state(gwc)
        @warn "The environment is in an absorbing state. This `step!` will not do anything. Please call `reset!`."
        gwc.reward = 0
    else
        step!(gwc.gw, a; rng=rng)
        s = state(gwc.gw)
        gwc.state .= factored_representation(gwc, s)
        gwc.reward = reward(gwc.gw)
    end
    nothing
end

in_absorbing_state(gwc::GridWorldContinuous) = in_absorbing_state(gwc.gw)


function print_policy(gwc::GridWorldContinuous{T}, p::AbstractPolicy{Vector{T}, Int}) where T
    states = map(s -> factored_representation(gwc, s), 1:length(gwc.gw.grid))
    meanings = map(a->action_meaning(gwc, a), p.(states))
    display(reshape(meanings, size(gwc.gw.grid)))
end

