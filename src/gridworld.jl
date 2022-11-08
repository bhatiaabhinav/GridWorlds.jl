using MDPs
using StaticArrays
import MDPs: state_space, action_space, discount_factor, horizon, action_meaning, start_state_support, start_state_probability, transition_support, transition_probability, reward, is_absorbing

export GridWorld

const GRID_WORLD_ACTION_MEANINGS = ["noop", "N", "E", "S", "W"]
const GRID_WORLD_OPPOSITE_ACTIONS = [1, 4, 5, 2, 3]
const GRID_WORLD_90_LEFT_ACTIONS = [1, 5, 2, 3, 4]
const GRID_WORLD_90_RIGHT_ACTIONS = [1, 3, 4, 5, 2]
const ACTION_DIRECTIONS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]

mutable struct GridWorld <: AbstractMDP{Int, Int}
    grid::Matrix{Char} # 'S'= start, 'G'=Goal, ' ' = Free, 'O' = Obstacle
    enter_rewards::Dict{Char, Float64}
    leave_rewards::Dict{Char, Float64}
    failuremode_noop_probability::Dict{Char, Float64}
    failuremode_slip_probability::Dict{Char, Float64}
    absorbing_states::Set{Char}

    # cached:
    const d₀_support::Vector{Int}
    const d₀::Vector{Float64}
    const T_support::Matrix{Vector{Int}}  # T_support(a | s)
    const T::Array{Float64, 3} # Pr(s'| a, s)
    const R::Array{Float64, 3}  # R(s′, a, s)

    state::Int
    action::Int
    reward::Float64

    function GridWorld(grid, enter_rewards::Dict{Char, Float64}; leave_rewards::Dict{Char, Float64}=Dict{Char, Float64}(), failuremode_noop_probability::Dict{Char, Float64}=Dict{Char, Float64}(), failuremode_slip_probability::Dict{Char, Float64}=Dict{Char, Float64}(), absorbing_states::Set{Char}=Set('G'))
        gw = new(grid, enter_rewards, leave_rewards, failuremode_noop_probability, failuremode_slip_probability, absorbing_states)
        d₀_support = collect(filter(s -> isstart(gw, s), 1:length(grid)))
        d₀ = Array{Float64}(undef, length(grid))
        T_support = fill(Int[], 5, length(grid))
        T = Array{Float64}(undef, length(grid), 5, length(grid))
        R = Array{Float64}(undef, length(grid), 5, length(grid))
        for s in 1:length(state_space(gw))
            tile = grid[s]
            if !haskey(enter_rewards, tile)
                enter_rewards[tile] = 0
            end
            if !haskey(leave_rewards, tile)
                leave_rewards[tile] = 0
            end
            if !haskey(failuremode_noop_probability, tile)
                failuremode_noop_probability[tile] = 0
            end
            if !haskey(failuremode_slip_probability, tile)
                failuremode_slip_probability[tile] = 0
            end

            d₀[s] = isstart(gw, s) ? (1 / length(d₀_support)) : 0
            for a in 1:length(action_space(gw))
                T_support[a, s] = collect(_transition_support(gw, s, a))
                for s′ in 1:length(state_space(gw))
                    T[s′, a, s] = _transition_probability(gw, s, a, s′)
                    R[s′, a, s] = isgoal(gw, s) ? 0 : leave_rewards[grid[s]] + enter_rewards[grid[s′]]
                end
            end
        end
        @assert all(sum(T, dims=1) .≈ 1.0)
        return new(grid, enter_rewards, leave_rewards, failuremode_noop_probability, failuremode_slip_probability, absorbing_states, d₀_support, d₀, T_support, T, R, 0, 0, 0)
    end
end




state_space(gp::GridWorld) = IntegerSpace(length(gp.grid))
action_space(gp::GridWorld) = IntegerSpace(5)
action_meaning(gp::GridWorld, a::Int) = GRID_WORLD_ACTION_MEANINGS[a]
discount_factor(gw::GridWorld) = 0.99
horizon(::GridWorld) = typemax(Int)

@inline nrows(gw::GridWorld) = size(gw.grid, 1)
@inline ncols(gw::GridWorld) = size(gw.grid, 2)
rcindex(gw::GridWorld, i::Int)::Tuple{Int, Int} = ((i - 1) % size(gw.grid, 1) + 1), ((i - 1) ÷ size(gw.grid, 1) + 1)
iindex(gw::GridWorld, rc::Tuple{Int, Int})::Int = (rc[2] - 1) * size(gw.grid, 1) + rc[1]

function start_state_support(gw::GridWorld)
    return gw.d₀_support
end

function start_state_probability(gw::GridWorld, s::Int)
    return gw.d₀[s]
end

function transition_support(gw::GridWorld, s::Int, a::Int)
    return gw.T_support[a, s]
end

function transition_probability(gw::GridWorld, s::Int, a::Int, s′::Int)
    return gw.T[s′, a, s]
end

function _transition_support(gw::GridWorld, s::Int, a::Int)
    if isgoal(gw, s) || isblocked(gw, s)  # absorbing states
        return [s]
    else
        return get_orthogonal_neighours(gw, s; include_self=true)
    end
end

function _transition_probability(gw::GridWorld, s::Int, a::Int, s′::Int)::Float64
    @assert s ∈ state_space(gw)
    @assert a ∈ action_space(gw)
    @assert s′ ∈ state_space(gw)

    if isgoal(gw, s)
        return Float64(s′ == s) # it's an absorbing state
    end

    if isblocked(gw, s)
        return Float64(s′ == s)  # s is a deadend state (but it's impossible to go there).
    end

    if isblocked(gw, s′)
        return 0  # don't allow moving to a blocked state from another state
    end


    if !isorthogonalneighbour(gw, s, s′)
        return 0
    else
        if a == 1 # NOOP
            return Float64(s′ == s)  # NOOP is always successful. All other states have zero probability.
        else
            move_dir_state = hypothetical_nextstate(gw, s, a)
            opp_dir_state = hypothetical_nextstate(gw, s, GRID_WORLD_OPPOSITE_ACTIONS[a])
            left_dir_state = hypothetical_nextstate(gw, s, GRID_WORLD_90_LEFT_ACTIONS[a])
            right_dir_state = hypothetical_nextstate(gw, s, GRID_WORLD_90_RIGHT_ACTIONS[a])

            if s′ == move_dir_state
                return isblocked(gw, s′) ? 0 : (1 - gw.failuremode_noop_probability[gw.grid[s]] - gw.failuremode_slip_probability[gw.grid[s]])
            elseif s′ == opp_dir_state
                return 0
            elseif (s′ == left_dir_state || s′ == right_dir_state)
                return isblocked(gw, s′) ? 0 : gw.failuremode_slip_probability[gw.grid[s]] / 2
            elseif s′ == s
                return gw.failuremode_noop_probability[gw.grid[s]] + (Float64(isnothing(left_dir_state) || isblocked(gw, left_dir_state)) + Float64(isnothing(right_dir_state) || isblocked(gw, right_dir_state))) * gw.failuremode_slip_probability[gw.grid[s]] / 2 + Float64(isnothing(move_dir_state) || isblocked(gw, move_dir_state)) * (1 - gw.failuremode_noop_probability[gw.grid[s]] - gw.failuremode_slip_probability[gw.grid[s]])
            end
        end
    end
end

function reward(gw::GridWorld, s::Int, a::Int, s′::Int)
    return gw.R[s′, a, s]
end

function is_absorbing(gw::GridWorld, s::Int)
    return isgoal(gw, s)
end


function print_policy(gw::GridWorld, p::AbstractPolicy{Int, Int})
    pstring = map(a->action_meaning(gw, a), p.(1:length(gw.grid)))
    display(reshape(pstring, size(gw.grid)))
end

#  --------------------- util functions -------------------------------
function isdiagneighbour(gw::GridWorld, s1::Int, s2::Int; include_self=true)::Bool
    rc1, rc2 = rcindex(gw.grid, s1), rcindex(gw.grid, s2)
    return all(-1 .<= (rc1 .- rc2) .<= 1) && sum(abs.(rc1 .- rc2)) == 2 && (include_self || s1 != s2)
end

function isorthogonalneighbour(gw::GridWorld, s1::Int, s2::Int; include_self=true)::Bool
    rc1, rc2 = rcindex(gw, s1), rcindex(gw, s2)
    if include_self
        return sum(abs.(rc1 .- rc2)) <= 1
    else
        return sum(abs.(rc1 .- rc2)) == 1
    end
end

function num_neighbors(gw::GridWorld, s::Int; include_self=true)::Int
    r, c = rcindex(gw, s)
    num_vertneighbors = length(max(r-1, 1):min(r+1, nrows(gw)))
    num_horiheighbors = length(max(c-1,1):min(c+1, ncols(gw)))
    # if gw.allow_diagonal_moves
    #     num_neighbors = num_vertneighbors * num_horiheighbors - 1
    # else
    num_neighbors = num_vertneighbors +  num_horiheighbors - 2
    # end
    if include_self
        num_neighbors += 1
    end
    return num_neighbors
end


function get_orthogonal_neighours(gw::GridWorld, s::Int; include_self=true)::Vector{Int}
    allow_diag = false
    r, c = rcindex(gw, s)
    neighbors = zeros(Int, num_neighbors(gw, s; include_self=include_self))
    neighbor_counter = 0
    for cn in max(c-1,1):min(c+1, ncols(gw))
        for rn in max(r-1, 1):min(r+1, nrows(gw))
            sn = iindex(gw, (rn, cn))
            if (include_self || sn != s) && (allow_diag || isorthogonalneighbour(gw, s, sn))
                neighbor_counter += 1
                neighbors[neighbor_counter] = sn
            end
        end
    end
    @assert neighbor_counter == length(neighbors)
    return neighbors
end

function hypothetical_nextstate(gw::GridWorld, s::Int, a::Int)::Union{Int, Nothing}
    rc = rcindex(gw, s)
    rc′ = rc .+ ACTION_DIRECTIONS[a]
    if all(1 .<= rc′ .<= size(gw.grid))
        return iindex(gw, rc′)
    else
        return nothing
    end
end


isgoal(gw::GridWorld, s::Int)::Bool = gw.grid[s] ∈ gw.absorbing_states
isstart(gw::GridWorld, s::Int)::Bool = gw.grid[s] == 'S'
isblocked(gw::GridWorld, s::Int)::Bool = gw.grid[s] == 'O' 

# ---------------------------------------------------------



