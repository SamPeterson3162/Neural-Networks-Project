using VortexLattice # load package we will be using
using DelimitedFiles
using Random
using Plots
using Lux
using JLD2

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
    ns = 10 # number of spanwise panels
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

# This must match your training file exactly.
# Ideally, put this in a shared file like "VLMModel.jl" and include() it in both,
# but for now, just copy-paste the definition.
function build_vlm_model(model_vars)
    inputs, outputs, ml = model_vars
    return Lux.Chain(
        Lux.Dense(inputs => ml, relu),
        Lux.Dense(ml => ml, relu),
        Lux.Dense(ml => ml, relu),
        Lux.Dense(ml => ml, relu),
        Lux.Dense(ml => ml, relu),
        Lux.Dense(ml => outputs) 
    )
end

# 2. LOAD THE SAVED DATA
println("Loading model weights...")
data = load("vlm_neural_net/models/10_panels_v2.0.jld2") # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MODEL 
ps_loaded = data["ps"]
st_loaded = data["st"]
vlm_norm = data["vlm_norm"] # Don't forget this!
model_vars = data["model_vars"]
inputs, outputs, ml = model_vars

model = build_vlm_model(model_vars)
# We don't need to initialize random weights because we are loading trained ones
# but Lux requires a placeholder setup call to get the structure ready if needed.
# However, since we have ps_loaded and st_loaded, we can often skip 'Lux.setup' 
# if we are sure the structure is identical.
# To be safe, let's just use the model structure we defined.

function predict_cl_cd(inputs::Vector{Float32})
    # inputs should be length 11: [Alpha, Root, x1, ..., x4, t0, ..., t4]
    
    # Note: vlm_norm rows align with your variables. 
    # Based on your get_data code, inputs are at indices (num_outputs+1) to end.
    # Let's assume inputs correspond to indices 49 to 59 (since outputs=48).
    
    num_outputs = outputs
    inputs_norm = zeros(Float32, length(inputs))
    
    for i in 1:length(inputs)
        # map input index 1 -> vlm_norm index 49
        norm_index = i + num_outputs 
        mean_val = vlm_norm[norm_index, 1]
        std_val = vlm_norm[norm_index, 2]
        inputs_norm[i] = (inputs[i] - mean_val) / std_val
    end
    # Lux returns (prediction, state). We only care about prediction here.
    preds_norm, _ = Lux.apply(model, inputs_norm, ps_loaded, st_loaded)

    # The model outputs normalized values. We need to scale them back up.
    preds_real = zeros(Float32, length(preds_norm))
    pred_length = length(preds_norm)
    for i in 1:pred_length
        mean_val = vlm_norm[i, 1] # Outputs are the first 48 rows
        std_val = vlm_norm[i, 2]
        preds_real[i] = (preds_norm[i] * std_val) + mean_val
    end

    return preds_real
end

function main()
    n = 1
    # Define function to get chords
    span_points = 11
    n_edges = 40 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< This will be four times spanwise panels used for VLM, which is 1 less than span_points
    input_lst = zeros(Float32,span_points *2 + 1,n)
    c_mat = zeros(Float32,span_points, n)
    t_mat = zeros(Float32,span_points, n)
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
    data_lst = zeros(n_edges, 2)
    a = input_lst[1,1]
    println("VLM run time:")
    cf,root = @time analyze_system(a, c_mat[:,1], t_mat[:,1])
    cf_x = cf[1][1:3:end]# extract coordinate coefficients
    cf_y = cf[1][2:3:end]# extract coordinate coefficients
    cf_z = cf[1][3:3:end]# extract coordinate coefficients
    input_lst[2:12] = input_lst[2:12] * root
    println("NN run time:")
    outputs = @time predict_cl_cd(input_lst[:,1])
    data_lst[1:Int64(n_edges/2),1] = cf_z[:]
    data_lst[Int64(n_edges/2+1):n_edges, 1] = cf_x[:]
    data_lst[1:end,2] = outputs[:]
    # Write to file
    plt = plot(range(-1, 1, length = Int64(n_edges/2)),data_lst[Int64(n_edges/2+1):end,1],label="VLM",title="drag")
    plot!(range(-1, 1, length = Int64(n_edges/2)),data_lst[Int64(n_edges/2+1):end,2],label="NN")
    output_file = "vlm_neural_net/vlm_model_test.data"
    display(plt)
    plt = plot(range(-1, 1, length = Int64(n_edges/2)),data_lst[1:Int64(n_edges/2),1],label="VLM",title="lift")
    plot!(range(-1, 1, length = Int64(n_edges/2)),data_lst[1:Int64(n_edges/2),2],label="NN")
    display(plt)
    
    delimiter = ' ' 
    writedlm(output_file, transpose(data_lst), delimiter)

    println("File '$output_file' written successfully.")
end

main()
