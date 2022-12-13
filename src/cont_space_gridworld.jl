using Random
using MDPs
import MDPs: state_space, action_space, action_meaning, reset!, step!, in_absorbing_state

export GridWorldContinuous

mutable struct GridWorldContinuous{T} <: AbstractMDP{Vector{T}, Int}
    gw::GridWorld

    const 𝕊::VectorSpace{T}

    state::Vector{T}
    action::Int
    reward::Float64

    function GridWorldContinuous{T}(gw::GridWorld) where T<:AbstractFloat
        rows, cols = size(gw.grid)
        𝕊 = VectorSpace{T}(T[1/rows, 1/cols], T[1, 1])
        new{T}(gw, 𝕊, zeros(T, 2), 0, 0.0)
    end
end

function translate_to_cont_state(gwc::GridWorldContinuous, s::Int)::Tuple{Float64, Float64}
    rcindex(gwc.gw, s) ./ size(gwc.gw.grid)
end

@inline state_space(gwc::GridWorldContinuous) = gwc.𝕊
@inline action_space(gwc::GridWorldContinuous) = action_space(gwc.gw)
@inline action_meaning(gwc::GridWorldContinuous, a::Int) = action_meaning(gwc.gw, a)

function reset!(gwc::GridWorldContinuous{T}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where T
    reset!(gwc.gw; rng=rng)
    gwc.state .= translate_to_cont_state(gwc, state(gwc.gw))
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
        gwc.state .= translate_to_cont_state(gwc, state(gwc.gw))
        gwc.reward = reward(gwc.gw)
    end
    nothing
end

in_absorbing_state(gwc::GridWorldContinuous) = in_absorbing_state(gwc.gw)


function print_policy(gwc::GridWorldContinuous{T}, p::AbstractPolicy{Vector{T}, Int}) where T
    states = map(s -> [T.(translate_to_cont_state(gwc, s))...], 1:length(gwc.gw.grid))
    meanings = map(a->action_meaning(gwc, a), p.(states))
    display(reshape(meanings, size(gwc.gw.grid)))
end