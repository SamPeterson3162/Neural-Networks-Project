using VortexLattice # load package we will be using
using DelimitedFiles
using Random

# This file writes our vlm_data file that will be used to train the neural network

function get_area(x1, x2, x3, x4)
    area = 1/2 + x1 + x2 + x3 + x4/2
    return area*2.5
end

function analyze_system(x1, x2, x3, x4)
    #Set up the geometric properties of half the wing'
    total_area = 20
    yle = [0.0, 2.5, 5, 7.5, 10] # leading edge y-position
    zle = [0.0, 0.0, 0.0, 0.0, 0.0] # leading edge z-position
    c_norm = total_area / get_area(x1, x2, x3, x4)
    chord = [1, x1, x2, x3, x4]*c_norm # chord length
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
    alpha = 5*pi/180 # angle of attack, where i is degrees but the program uses radians.
    beta = 0.0 # sideslip angle
    Omega = [0.0, 0.0, 0.0] # rotational velocity around the reference location
    fs = Freestream(Vinf, alpha, beta, Omega)
    # we already mirrored, so we do not need a symmetric calculation
    symmetric = false

    steady_analysis!(system, ref, fs; symmetric)
    # Extract all body forces
    CF, CM = body_forces(system; frame=Wind())
    # extract aerodynamic forces
    CD, CY, CL = CF
    Cl, Cm, Cn = CM
    CDiff = far_field_drag(system)
    return CL, CDiff, c_norm
end

function main()
    n = 500
    # Define function to get chords
    input_lst = zeros(6,n)
    input_lst[2:6,1] .= 1, .9375,.75, .4375, .01
    for i in 2:n # Uses random to get random chord values that are a percentage of the adjacent chord
        x_root = 1
        x1 = x_root * rand(40:100)/100 
        x2 = x1 * rand(20:100)/100
        x3 = x2 * rand(10:100)/100
        x4 = x3 * rand(5:100)/100
        input_lst[2:6,i] .= x_root, x1, x2, x3, x4
    end
    for i in 1:n
        x1, x2, x3, x4 = input_lst[3:6,i]
        CL, CDiff, root = analyze_system(x1, x2, x3, x4)
        input_lst[:,i] .= input_lst[:,i] .* root
        input_lst[1,i] = CL/CDiff
        input_lst[2,i] = root
    end
    # Write to file
    output_file = "vlm_neural_net/vlm_data_file.data"
    delimiter = ' ' 
    writedlm(output_file, transpose(input_lst), delimiter)

    println("File '$output_file' written successfully.")
end