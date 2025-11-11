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

function analyze_system(a,root,x1, x2, x3, x4)
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
    # Extract all body forces
    CF, CM = body_forces(system; frame=Wind())
    # extract aerodynamic forces
    CD, CY, CL = CF
    Cl, Cm, Cn = CM
    CDiff = far_field_drag(system)
    return CL, CDiff
end

function get_data(file) # Reads, processes, and splits our data from the vlm file into sets for training and testing
    println("Reading File") # Read data from our vlm file
    lines = readlines(file)
    num_lines = length(lines)
    vlm_data = zeros(8, num_lines)
    for i in 1:num_lines #Split the data up into 8 columns within a matrix
        line = split(lines[i])
        for j in 1:8
            vlm_data[j,i] = parse(Float32,line[j])
        end
    end
    # vlm_data is ordered with CL, CD, Alpha, Root, x1, x2, x3, x4
    # Define test and trainining sets
    println("Defining vectors of data")
    ratio = 0.8 # We want 80 percent of our data set used for training, the rest for testing
    train_size = round(Int, num_lines * ratio)
    test_size = num_lines - train_size
    training_vars = zeros(6,train_size) # Create zero lists for training and testing data sets
    training_vlm = zeros(2,train_size)
    test_vars = zeros(6,test_size)
    test_vlm = zeros(2,test_size)
    # Normalize Data
    println("Normalizing the data")
    vlm_norm = zeros(8, 2) # 1-8 for each variable within the data, 1-2 for mean and standard deviation respectively
    for i in 1:8 # Iterates through each variable and adds mean and standard deviation to vlm_norm matrix
        c_mean = mean(vlm_data[i,:])
        c_std = std(vlm_data[i,:])
        vlm_data[i,:] .= (vlm_data[i,:].-c_mean)./c_std
        # Save mean and standard deviation for vlm data set
        vlm_norm[i,1] = c_mean
        vlm_norm[i,2] = c_std
    end
    # Split vlm_data using the 80/20 split for training and testing
    println("Defining training and testing sets")
    indices = shuffle!(collect(1:num_lines)) # Randomizes order of indices
    train_ind = indices[1:train_size]
    test_ind = indices[train_size+1:end]
    for i in 1:train_size
        training_vlm[1:2,i] = vlm_data[1:2,train_ind[i]] # adds values from selected indices to training set outputs
        for j in 3:8
            training_vars[j-2,i] = vlm_data[j,train_ind[i]] # adds values from selected indices to training set inputs
        end
    end
    for i in 1:num_lines - train_size
        test_vlm[1:2,i] = vlm_data[1:2,test_ind[i]] # adds values from selected indices to testing set outputs
        for j in 3:8
            test_vars[j-2,i] = vlm_data[j,test_ind[i]] # adds values from selected indices to testing set inputs
        end
    end
    return training_vars, training_vlm, test_vars, test_vlm, vlm_norm
end

function train()
    # Extract data from vlm_alpha_data_file
    training_vars, training_vlm, test_vars, test_vlm, vlm_norm = get_data("vlm_neural_net/vlm_alpha_data_file.data")
    #region Model Definition
    println("Defining Model")
    model = Lux.Chain(
        Lux.Dense(6 => 30,relu),
        Lux.Dense(30 => 30,relu),
        Lux.Dense(30 => 30,relu),
        Lux.Dense(30 => 30,relu),
        Lux.Dense(30 => 30,relu),
        Lux.Dense(30 => 2)
    )
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    epochs = 10000 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DEFINE EPOCHS HERE
    # __________________________________________________________________________________________________________
    train_losses = ones(epochs) # Create local lists for training and testing losses
    test_losses = ones(epochs)
    cl_losses = ones(epochs)
    cd_losses = ones(epochs)
    #endregion
    #region Loss Functions
    # Define loss function using Mean Squared
    println("Defining loss function")
    function loss(model, ps, st, x, y) #= Loss function that deals with both CL and CD, returning losses for each as well as the combined
        loss used for optimization in our model=#
        true_y, new_state = model(x, ps, st)
        CL_pred = true_y[1,:] # Get just CL values
        CD_pred = true_y[2,:] # Get just CD values
        l_CL = sum((CL_pred.-y[1,:]).^2)/size(y, 2) # Compute Mean Standard Error for each seperately
        l_CD = sum((CD_pred.-y[2,:]).^2)/size(y, 2)
        l = l_CL + l_CD # Adds Errors together, values here are still normalized
        return l, l_CL, l_CD, new_state
    end
    function abs_loss(model, ps, st, x, y)
        true_y, new_state = model(x, ps, st)
        # Denormalize our values for CL and CD to provide true errors
        true_y2 = zeros(size(true_y))
        y2 = zeros(size(y))
        print(size(y))
        print(size(true_y))
        true_y2[1, :] .= true_y[1,:].*vlm_norm[1,2] .+vlm_norm[1,1] # predicted CL Values being denormalized
        true_y2[2, :] .= true_y[2,:].*vlm_norm[2,2] .+vlm_norm[2,1] # predicted CD Values being denormalized
        y2[1,:] .= y[1,:].*vlm_norm[1,2] .+vlm_norm[1,1] # Actual CL values being denormalized
        y2[2,:] .= y[2,:].*vlm_norm[2,2] .+vlm_norm[2,1] # Actual CD values being denormalized
        l_cl = sum(abs.(true_y2[1,:].-y2[1,:]))/size(y, 2) # Compute MSE for CL
        l_cd = sum(abs.(true_y2[2,:].-y2[2,:]))/size(y, 2) # Compute MSE for CD
        return l_cl, l_cd, new_state
    end
    #endregion
    #region Optimization
    optimizer = Optimisers.ADAMW(0.0005f0, (0.9f0, 0.999f0), 0.1f0) # Using ADAMW as decay helps increase generalization and reduce test loss
    opt_state = Optimisers.setup(optimizer, ps) # Set states
    # Run iterations with epochs
    println("Training in progress...")
    batch_size = 64 # <<<<<<<<<<<<<<<<<<<<<<<<<< BATCH SIZE HERE
    loss_val = 1 
    cl_loss = 1 # create local loss values to update in for loop
    cd_loss = 1 # '                                             '
    final_cl_error, final_cd_error = 0, 0 # '                   '
    loader = DataLoader((training_vars, training_vlm), batchsize=batch_size, shuffle=true) # Set up batch loading
    for epoch in 1:epochs # Iterates through epochs iterations
        local epoch_loss = 0.0 # Defines loss of this epoch iteration
        local num_batches = 0 
        for (x_batch, y_batch) in loader
            (loss_val, cl_loss, cd_loss, updated_state), grads = Zygote.withgradient(
                p -> loss(model, p, st, x_batch, y_batch), # Use the batch
                ps
            )

            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
            st = updated_state # Update state and other variables
            epoch_loss += loss_val
            num_batches += 1
        end
        current_loss = loss_val
        #train_losses[epoch] = current_loss
        train_losses[epoch] = epoch_loss / num_batches # Used for plotting
        test_loss_val, _ = loss(model, ps, st, test_vars, test_vlm) 
        test_losses[epoch] = test_loss_val # Used for plotting loss, also printed after every 100 epochs
        cl_losses[epoch] = cl_loss # Used for plotting loss
        cd_losses[epoch] = cd_loss # Used for plotting loss
        if epoch == epochs
            final_cl_error, final_cd_error, _ = abs_loss(model, ps, st, test_vars, test_vlm) # Save final absolute errors for cl and cd
        end
        if epoch % 100 == 0
            println("\ncurrent test loss: $test_loss_val") # Print test loss every 100 epochs
        end
    end
    #endregion
    #region Loss Plotting
    loss_plt = plot(1:epochs,train_losses,label="Training Set Loss",title="Loss over time",y_scale=:log10)
    plot!(1:epochs,test_losses,label="Testing Set Loss")
    plot!(1:epochs,cl_losses,label="Lift Loss")
    plot!(1:epochs,cd_losses,label="Drag Loss")
    savefig(loss_plt, "vlm_neural_net/vlm_Loss_Function.png")
    #endregion
    return model, ps, st, vlm_norm, final_cl_error, final_cd_error
end

function main()
    model, ps, st, vlm_norm, cl_error, cd_error = train() # run train function to train the neural network
    println("Test Set CL Average Absolute Error: $cl_error")
    println("Test Set CD Average Absolute Error: $cd_error")
    #region User Testing Values
    # Apply the model to a random line of data
    prediction(x) = Lux.apply(model, x, ps, st)
    test_vals = zeros(6,1)
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
        test_vals[:,1] .= 5, 1, x1, x2, x3, x4
        test_vals[2:6,1] .= test_vals[2:6,1] .* 8 / (1/2 + x1 + x2 + x3 + x4/2) # 8 is derived from the area of 20 divided by the distance between chords of 2.5
        test_vals2 = test_vals .* 1 # To use for regular vlm analysis
        for i in 1:6
            test_vals[i,1] = (test_vals[i,1] - vlm_norm[i+2,1]) / vlm_norm[i+2,2]
        end
        n_vlm_val, _ = prediction(test_vals)
        vlm_val = [0.0, 0.0]
        vlm_val[1] = n_vlm_val[1]*vlm_norm[1,2] + vlm_norm[1,1]
        vlm_val[2] = n_vlm_val[2]*vlm_norm[2,2] + vlm_norm[2,1]

        CL = vlm_val[1]
        CD = vlm_val[2]
        alpha = 5
        CL1, CDiff = analyze_system(alpha,test_vals2[2,1],test_vals2[3,1],test_vals2[4,1],test_vals2[5,1],test_vals2[6,1])
        println("")
        println("Predicted CL Value:                 $CL")
        println("Actual CL Value:                    $CL1")
        println("Case CL error:                      $(abs(CL-CL1))")
        println("Test Set CL Average Absolute Error: $cl_error")
        println("")
        println("Predicted CD Value:                 $CD")
        println("Actual CD Value:                    $CDiff")
        println("Case CD error:                      $(abs(CD-CDiff))")
        println("Test Set CD Average Absolute Error: $cd_error")
    end
    #endregion
end


