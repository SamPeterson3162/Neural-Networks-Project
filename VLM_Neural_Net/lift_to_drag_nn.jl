# Load necessary Packages
using Lux
using Plots
using Optimisers
using Zygote
using Random
using Statistics
using MLUtils
using VortexLattice

# This file trains a neural network with the chord values and CL/CDiff values given by lift_to_drag.jl and then
# Lets us compute the value using the neural net for any distribution of chord lengths and gives the true value

function analyze_system(x1, x2, x3, x4)
    # This is the same function used to create the file, but here it is just used to test the final value input by the user
    xle = [0.0, 2-x1/2, 2-x2/2, 2-x3/2, 2-x4/2] # leading edge x-position
    yle = [0.0, 2.5, 5, 7.5, 10] # leading edge y-position
    zle = [0.0, 0.0, 0.0, 0.0, 0.0] # leading edge z-position
    chord = [4, x1, x2, x3, x4] # chord length
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
    return CL, CDiff
end

function get_data(file)
    println("Reading File")
    lines = readlines(file)
    num_lines = length(lines)
    vlm_data = zeros(6, num_lines)
    for i in 1:num_lines
        line = split(lines[i])
        for j in 1:6
            vlm_data[j,i] = parse(Float32,line[j])
        end
    end
    # Define test and trainining sets
    println("Defining vectors of data")
    ratio = 0.8
    train_size = round(Int, num_lines * ratio)
    test_size = num_lines - train_size
    training_vars = zeros(5,train_size)
    training_vlm = zeros(1,train_size)
    test_vars = zeros(5,test_size)
    test_vlm = zeros(1,test_size)
    # Normalize Data
    println("Normalizing the data")
    vlm_norm = zeros(6, 2)
    for i in 1:6
        c_mean = mean(vlm_data[i,:])
        c_std = std(vlm_data[i,:])
        vlm_data[i,:] .= (vlm_data[i,:].-c_mean)./c_std
        # Save mean and standard deviation for vlm data set
        vlm_norm[i,1] = c_mean
        vlm_norm[i,2] = c_std
    end
    # Split vlm_data using the 80/20 split for training and testing
    println("Defining training and testing sets")
    indices = shuffle!(collect(1:num_lines))
    train_ind = indices[1:train_size]
    test_ind = indices[train_size+1:end]
    for i in 1:train_size
        training_vlm[1,i] = vlm_data[1,train_ind[i]]
        for j in 2:6
            training_vars[j-1,i] = vlm_data[j,train_ind[i]]
        end
    end
    for i in 1:num_lines - train_size
        test_vlm[1,i] = vlm_data[1,test_ind[i]]
        for j in 2:6
            test_vars[j-1,i] = vlm_data[j,test_ind[i]]
        end
    end
    return training_vars, training_vlm, test_vars, test_vlm, vlm_norm
end

function train()
    # Extract data from auto-vlm.data file
    training_vars, training_vlm, test_vars, test_vlm, vlm_norm = get_data("vlm_neural_net/vlm_data_file.data")
    # Define our model
    println("Defining Model")
    num_samples = size(training_vars,2)
    model = Lux.Chain(
        Lux.Dense(5 => 25,relu),
        Lux.Dense(25 => 10,relu),
        Lux.Dense(10 => 1)
    )
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    epochs = 10000
    train_losses = ones(epochs)
    test_losses = ones(epochs)
    # Define loss function using Mean Squared
    println("Defining loss function")
    function loss(model, ps, st, x, y)
        true_y, new_state = model(x, ps, st)
        l = sum((true_y.-y).^2)/size(y, 2)
        return l, new_state
    end
    function abs_loss(model, ps, st, x, y)
        true_y, new_state = model(x, ps, st)
        # Denormalize
        true_y2 = zeros(size(true_y))
        y2 = zeros(size(y))
        true_y2 .= true_y.*vlm_norm[1,2] .+vlm_norm[1,1]
        y2 .= y.*vlm_norm[1,2] .+vlm_norm[1,1]
        l = sum(abs.(true_y2.-y2))/size(y, 2)

        return l, new_state
    end
    optimizer = Optimisers.ADAM(0.00005f0) # Didn't use the ADAMW optimizer here as it did fine without it
    opt_state = Optimisers.setup(optimizer, ps)
    # Run iterations with epochs
    println("Training in progress...")
    batch_size = 64
    loss_val = 1
    loader = DataLoader((training_vars, training_vlm), batchsize=batch_size, shuffle=true)
    final_error = 0
    for epoch in 1:epochs # Iterates through epochs iterations
        local epoch_loss = 0.0
        local num_batches = 0
        for (x_batch, y_batch) in loader
            (loss_val, updated_state), grads = Zygote.withgradient(
                p -> loss(model, p, st, x_batch, y_batch), # Use the batch
                ps
            )

            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
            st = updated_state
            epoch_loss += loss_val
            num_batches += 1
        end
        current_loss = loss_val
        #train_losses[epoch] = current_loss
        train_losses[epoch] = epoch_loss / num_batches
        test_loss_val, _ = loss(model, ps, st, test_vars, test_vlm)
        test_losses[epoch] = test_loss_val
        if epoch == epochs
            final_error, _ = abs_loss(model, ps, st, test_vars, test_vlm)
        end
        if epoch % 1000 == 0
            println("\ncurrent test loss: $test_loss_val")
        end
    end
    loss_plt = plot(1:epochs,train_losses,label="Training Set Loss",title="Loss over time",y_scale=:log10)
    plot!(1:epochs,test_losses,label="Testing Set Loss")
    savefig(loss_plt, "vlm_neural_net/vlm_Loss_Function.png")

    return model, ps, st, vlm_norm, final_error
end

function main()
    model, ps, st, vlm_norm, final_error = train()
    println("Test Set Average Absolute Error: $final_error")
    # Apply the model to a random line of data
    prediction(x) = Lux.apply(model, x, ps, st)
    test_vals = zeros(4,1)
    while true
        println("Would you like to compute the coefficient of lift over drag for a chord distribution? (Y/N)")
        answer = readline()
        if answer == "N" || answer == "n"
            break
        end
        println("Input values for chord along the span ranging from 0 to 1 where 1 is the root chord")
        println("These chord values will correspond to the quarter span, half span, 3-quarter span, and tip chords")
        print("x1:")
        x1 = parse(Float32, readline())
        print("x2:")
        x2 = parse(Float32, readline())
        print("x3:")
        x3 = parse(Float32, readline())
        print("x4:")
        x4 = parse(Float32, readline())
        test_vals[:,1] .= x1*4, x2*4, x3*4, x4*4
        test_vals2 = test_vals .* 1 # To use for regular vlm analysis
        for i in 2:5
            test_vals[i-1,1] = (test_vals[i-1,1] - vlm_norm[i,1]) / vlm_norm[i,2]
        end
        n_vlm_val, _ = prediction(test_vals)
        vlm_val = n_vlm_val.*vlm_norm[1,2] .+ vlm_norm[1,1]
        vlm_val = vlm_val[1]
        println("Predicted VLM Value: $vlm_val")
        CL, CDiff = analyze_system(test_vals2[1,1],test_vals2[2,1],test_vals2[3,1],test_vals2[4,1])
        final_vlm = CL/CDiff
        println("Actual VLM Value: $final_vlm")
    end
end


