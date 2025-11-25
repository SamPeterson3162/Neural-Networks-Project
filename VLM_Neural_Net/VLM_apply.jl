using VortexLattice # load package we will be using
using DelimitedFiles
using Random

# This file writes our vlm_data file that will be used to train the neural network

function get_area(c_vec)
    area = c_vec[1]/2 + sum(c_vec[2:end-1]) + c_vec[end]/2
    return area*2.5
end

function analyze_system(a, c_vec, t_vec)
    #Set up the geometric properties of half the wing'
    span_points = size(c_vec,1)
    total_area = 5*(span_points-1)
    yle = range(0,10, length=span_points) # leading edge y-position
    zle = zeros(span_points) # leading edge z-position
    c_norm = total_area / get_area(c_vec)
    chord = c_vec[1:end]*c_norm # chord length
    xle = ones(span_points)* chord[1] - chord # leading edge x-position
    theta = t_vec[1:end] * pi/180 # twist (in radians)
    phi = zeros(span_points) # section rotation about the x-axis
    fc = fill((xc) -> 0, span_points) # camberline function for each section (y/c = f(x/c))
    # define the number of panels in the spanswise and chordwise directions
    ns = 20 # number of spanwise panels
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
    # Extract all body forces
    CF, CM = body_forces(system; frame=Wind())
    # extract aerodynamic forces
    CD, CY, CL = CF
    Cl, Cm, Cn = CM
    CDiff = far_field_drag(system)
    cf, cm = lifting_line_coefficients(system; frame=Wind())
    return cf, c_norm
end

function main()
    n = 10000
    # Define function to get chords
    span_points = 20
    n_edges = 80 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< This will be four times spanwise panels used for VLM 
    input_lst = zeros(span_points *2 + 1,n)
    c_mat = zeros(span_points, n)
    t_mat = zeros(span_points, n)
    for i in 1:n
        x0 = 1
        c_mat[1,i] = x0
        c_max = 100
        for j in 2:span_points
            c_cent = round(Int64,(j-1)/(span_points-1)*100) # point along the chord as percentage of chord from root. 0-100
            x_j = x0 * rand(max(1,100-c_cent):c_max)
            c_mat[j,i] = x_j
            c_max = x_j
        end
        input_lst[1,i] = rand(1:100)/10
    end
    for i in 1:n
        for j in 1:span_points
            t_j = rand(-100:100)/50
            t_mat[j,i] = t_j
        end
    end
    input_lst[2:span_points+1,:] = c_mat[:,:]
    input_lst[span_points+2:end,:] = t_mat[:,:]
    data_lst = zeros(1+2*span_points+n_edges, n)
    for i in 1:n
        a = input_lst[1,i]
        cf, root = analyze_system(a, c_mat[:,i], t_mat[:,i])
        cf_x = cf[1][1:3:end]# extract coordinate coefficients
        cf_y = cf[1][2:3:end]# extract coordinate coefficients
        cf_z = cf[1][3:3:end]# extract coordinate coefficients
        data_lst[1:Int64(n_edges/2),i] = cf_z[:]
        data_lst[Int64(n_edges/2+1):n_edges, i] = cf_x[:]
        data_lst[n_edges + 1,i] = a
        data_lst[n_edges + 2:end-span_points,i] .= input_lst[2:span_points+1,i] .* root
        data_lst[end-span_points+1:end,i] .= input_lst[span_points+2:end,i]
        if i % round(n/100) == 0
            cent = round(i*100/n;digits=2)
            println("$cent% written")
        end
    end
    # Write to file
    output_file = "vlm_neural_net/vlm_data_file.data"
    delimiter = ' ' 
    writedlm(output_file, transpose(data_lst), delimiter)

    println("File '$output_file' written successfully.")
end