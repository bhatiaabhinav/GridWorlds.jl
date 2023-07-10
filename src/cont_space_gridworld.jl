using Random
using MDPs
import MDPs: state_space, state, action_space, action_meaning, reset!, step!, in_absorbing_state, unwrapped, start_state_support, start_state_probability, start_state_distribution, transition_support, transition_probability, transition_distribution, is_absorbing, visualize

export GridWorldContinuous

mutable struct GridWorldContinuous{T} <: AbstractWrapper{Vector{T}, Int}
    gw::GridWorld
    include_tile_type::Bool

    const 𝕊::EnumerableVectorSpace{T}

    state::Vector{T}

    tile_type_to_onehot::Dict{Char, Vector{T}}

    function GridWorldContinuous{T}(gw::GridWorld, include_tile_type::Bool=false) where T<:AbstractFloat
        tile_types = gw.enter_rewards |> keys
        ntiletypes = length(tile_types)
        tile_type_to_onehot = Dict{Char, Vector{T}}()
        for (i, tile_type) in enumerate(tile_types)
            tile_type_to_onehot[tile_type] = zeros(T, ntiletypes)
            tile_type_to_onehot[tile_type][i] = 1
        end
        elements = Vector{T}[]  # elements in state space
        for s in 1:length(gw.grid)
            if include_tile_type
                push!(elements, T[(rcindex(gw, s) ./ size(gw.grid))..., tile_type_to_onehot[gw.grid[s]]...])
            else
                push!(elements, T[(rcindex(gw, s) ./ size(gw.grid))...])
            end
        end
        𝕊 = EnumerableVectorSpace{T}(elements)
        s = 𝕊[1]
        new{T}(gw, include_tile_type, 𝕊, s, tile_type_to_onehot)
    end
end

@inline unwrapped(gwc::GridWorldContinuous) = gwc.gw
@inline state_space(gwc::GridWorldContinuous) = gwc.𝕊
@inline state(gwc::GridWorldContinuous) = gwc.state
start_state_support(gwc::GridWorldContinuous) = Iterators.map(s -> gwc.𝕊[s], start_state_support(gwc.gw))
start_state_probability(gwc::GridWorldContinuous{T}, s::Vector{T}) where {T} = start_state_probability(gwc.gw, indexin(s, gwc.𝕊)[])
start_state_distribution(gwc::GridWorldContinuous, support) = start_state_distribution(gwc.gw, map(s -> indexin(s, gwc.𝕊)[], support))
transition_support(gwc::GridWorldContinuous{T}, s::Vector{T}, a::Int) where {T} = transition_support(gwc.gw, indexin(s, gwc.𝕊)[], a)
transition_probability(gwc::GridWorldContinuous{T}, s::Vector{T}, a::Int, s′::Vector{T}) where {T} = transition_probability(gwc.gw, indexin(s, gwc.𝕊)[], a, indexin(s′, gwc.𝕊)[])
transition_distribution(gwc::GridWorldContinuous{T}, s::Vector{T}, a::Int, support) where {T} = transition_distribution(gwc.gw, indexin(s, gwc.𝕊)[], a, map(s′ -> indexin(s′, gwc.𝕊)[], support))
reward(gwc::GridWorldContinuous{T}, s::Vector{T}, a::Int, s′::Vector{T}) where {T} = reward(gwc.gw, indexin(s, gwc.𝕊)[], a, indexin(s′, gwc.𝕊)[])
is_absorbing(gwc::GridWorldContinuous{T}, s::Vector{T}) where {T} = is_absorbing(gwc.gw, indexin(s, gwc.𝕊)[])
visualize(gwc::GridWorldContinuous{T}, s::Vector{T}; kwargs...) where {T} = visualize(gwc.gw, indexin(s, gwc.𝕊)[], kwargs...)


function reset!(gwc::GridWorldContinuous{T}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where T
    reset!(gwc.gw; rng=rng)
    gwc.state .= gwc.𝕊[state(gwc.gw)]
    nothing
end

function step!(gwc::GridWorldContinuous, a::Int; rng=Random.AbstractRNG)::Nothing
    step!(gwc.gw, a; rng=rng)
    gwc.state .= gwc.𝕊[state(gwc.gw)]
    nothing
end


function print_policy(gwc::GridWorldContinuous{T}, p::AbstractPolicy{Vector{T}, Int}) where T
    states = collect(gwc.𝕊)
    meanings = map(a->action_meaning(gwc, a), p.(states))
    display(reshape(meanings, size(gwc.gw.grid)))
end

