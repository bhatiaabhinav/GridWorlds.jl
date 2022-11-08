export CliffWalkGridWorld

function CliffWalkGridWorld()
    grid = [
        ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ';
        ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ';
        ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ';
        'S' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'C' 'G'
    ]
    return GridWorld(grid, Dict('S' => -1.0, ' ' => -1.0, 'G' => -1.0, 'C' => -100.0); absorbing_states=Set(['C', 'G']))
end