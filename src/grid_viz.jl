import MDPs: visualize
using Luxor
using Luxor.Colors

# SCALE = 10
drawcolors=Dict('O' => "black", 'C' => "red")
drawtext=Dict('O' => "", 'C' => "")
action_draw_text = ["□", "↑", "→", "↓", "←"]

function visualize(gw::GridWorld, s::Int, a::Union{Nothing, Int}=nothing, rew::Union{Nothing, Int}=nothing, args...; value_fn::Union{Nothing, Dict{Vector{T}, <:Real}, Vector{<:Real}} = nothing, vmax::Real=1, filename::Union{Nothing, String}=nothing, show_action=true, kwargs...)::Matrix{ARGB32} where T
    if isnothing(a)
        a = max(gw.action, 1)
    end
    if isnothing(rew)
        rew = gw.reward
    end
    g = gw.grid

    R, C = size(g)

    SCALE = 1000 ÷ max(R + 1, C)
    if show_action
        W, H = SCALE * C, SCALE * (R + 1)
    else
        W, H = SCALE * C, SCALE * R
    end
    W = W % 2 != 0 ? W + 1 : W
    H = H % 2 != 0 ? H + 1 : H

    Drawing(W, H, isnothing(filename) ? :image : filename)
    origin(Point(0, 0))
    background("white")


    for (i, cell) in enumerate(g)
        r, c = rcindex(gw, i)
        x, y = c, r

        setline(0)
        setcolor(get(drawcolors, cell, "white"))
        rect(SCALE * (x - 1), SCALE * (y - 1), SCALE, SCALE, :fill)

        if !isnothing(value_fn)
            v = value_fn isa Dict ? get(value_fn, T[r/R, c/C], 0) : value_fn[i]
            higest_v = maximum(values(value_fn))
            # if higest_v > 99
                if abs(v) > 0.5
                    vround = round(v; sigdigits=2)
                    color = RGBA(v < 0, v > 0, 0, min(abs(v) / vmax, 1))
                    setcolor(color)
                    rect(SCALE * (x - 1), SCALE * (y - 1), SCALE, SCALE, :fill)
                    setcolor("black")
                    fontsize(0.3 * SCALE)
                    text("$vround", Point(SCALE * (x - 0.5), SCALE * (y - 0.5)), halign=:center, valign=:middle)
                end
            # end
        end

        setcolor("black")
        str = get(drawtext, cell, string(cell))
        fontsize(0.75 * SCALE)
        th = textextents(str)[4]
        text(str, Point(SCALE * (x - 0.5), SCALE * (y - 0.5) - th/2), halign=:center, valign=:top)

        setcolor("black")
        setline(SCALE / 20)
        rect(SCALE * (x - 1), SCALE * (y - 1), SCALE, SCALE, :stroke)
    end

    setcolor("black")
    r, c = rcindex(gw, s)
    x, y = c, r
    circle(SCALE * (x - 0.5), SCALE * (y - 0.5), SCALE / 4, :fill)


    setcolor("black")
    setline(SCALE / 10)
    rect(0, 0, C * SCALE, R * SCALE, :stroke)
    setcolor("white")
    setline(SCALE / 10)
    rect(0, R * SCALE, C * SCALE, SCALE, :fill)

    if show_action
        setcolor("black")
        fontsize(0.5 * SCALE)
        str = "Action: $(action_draw_text[a])   Reward: $(round(rew; sigdigits=2))"
        th = textextents(str)[4]
        text(str, Point(C * SCALE / 2, (R + 0.5) * SCALE - th/2), halign=:center, valign=:top)
    end

    img = image_as_matrix()

    !isnothing(filename) && finish()

    return img
end


function visualize(gw::GridWorldContinuous{T}, s::Vector{T}, args...; kwargs...) where T
    r, c = s .* size(gw.gw.grid)
    s = iindex(gw.gw, (Int(round(r)), Int(round(c))))
    visualize(gw.gw, s, args...; kwargs...)
end