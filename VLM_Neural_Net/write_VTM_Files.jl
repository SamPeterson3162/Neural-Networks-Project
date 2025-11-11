# Load necessary Packages
using Lux
using Plots
using Optimisers
using Zygote
using Random
using Statistics
using MLUtils
using VortexLattice

#= This file trains a neural network with the angle of attack and chord values to predict  CL and CD values given by lift_to_drag_alpha.jl and then
lets us compute the value using the neural net for any distribution of chord lengths and gives the true value=#

function analyze_system(i, a, root,x1, x2, x3, x4)
    # This is the same function used to create the file, but here it is just used to test the final value input by the user
    yle = [0.0, 2.5, 5, 7.5, 10] # leading edge y-position
    zle = [0.0, 0.0, 0.0, 0.0, 0.0] # leading edge z-position
    chord = [root, x1, x2, x3, x4] # chord length
    xle = ones(5)* chord[1] - chord # leading edge x-position
    theta = [0.0*pi/180, 0.0*pi/180, 0.0*pi/180, 0.0*pi/180, 0.0*pi/180] # twist (in radians)
    phi = [0.0, 0.0, 0.0, 0.0, 0.0] # section rotation about the x-axis
    fc = fill((xc) -> 0, 5) # camberline function for each section (y/c = f(x/c))
    # define the number of panels in the spanswise and chordwise directions
    ns = 12 # number of spanwise panels
    nc = 6  # number of chordwise panels
    spacing_s = Uniform() # spanwise discretization scheme
    spacing_c = Uniform() # chordwise discretization scheme
    # generate a lifting surface for the defined geometry
    grid, ratio = wing_to_grid(xle, yle, zle, chord, theta, phi, ns, nc;
    fc = fc, spacing_s=spacing_s, spacing_c=spacing_c, mirror=true)
    # combine all grids to one vector as well as ratios into one vector
    grids = [grid]
    ratios = [ratio]
    system = System(grids; ratios)
    # define freestream and reference for the model
    Sref = 30.0 # reference area
    cref = 2.0  # reference chord
    bref = 15.0 # reference span
    rref = [0.50, 0.0, 0.0] # reference location for rotations/moments (typically the c.g.)
    Vinf = 1.0 # reference velocity (magnitude)
    ref = Reference(Sref, cref, bref, rref, Vinf)
    # freestream definition
    # Insert range for angle in degrees below
    # Iterate through the different angles of attack, one degree at a time
    alpha = a*pi/180 # angle of attack, where i is degrees but the program uses radians.
    beta = 0.0 # sideslip angle
    Omega = [0.0, 0.0, 0.0] # rotational velocity around the reference location
    fs = Freestream(Vinf, alpha, beta, Omega)
    # we already mirrored, so we do not need a symmetric calculation
    symmetric = false
    steady_analysis!(system, ref, fs; symmetric)
    write_vtk("vlm_neural_net/VTM_Files/EG$i/Wing_EG_$i", system)
end

function get_data(file) # Reads, processes, and splits our data from the vlm file into sets for training and testing
    println("Reading File") # Read data from our vlm file
    lines = readlines(file)
    num_lines = 4
    vlm_data = zeros(8, num_lines)
    for i in 1:num_lines #Split the data up into 8 columns within a matrix
        line = split(lines[i])
        for j in 1:8
            vlm_data[j,i] = parse(Float32,line[j])
        end
    end
    # vlm_data is ordered with CL, CD, Alpha, Root, x1, x2, x3, x4
    Wing_1 = vlm_data[3:8,1]
    Wing_2 = vlm_data[3:8,2]
    Wing_3 = vlm_data[3:8,3]
    Wing_4 = vlm_data[3:8,4]
    Wings = [Wing_1, Wing_2, Wing_3, Wing_4]
    return Wings
end

function main()
    Wings = get_data("vlm_neural_net/vlm_alpha_data_file.data")
    for i in 1:4
        system = analyze_system(i, Wings[i][1], Wings[i][2], Wings[i][3], Wings[i][4], Wings[i][5], Wings[i][6])
    end
    println("VTM Files written successfully")
end


