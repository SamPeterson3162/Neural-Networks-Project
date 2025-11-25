# Load necessary Packages
using Lux
using Plots
using Optimisers
using Zygote
using Random
using Statistics
using MLUtils
using VortexLattice
using LinearAlgebra
using JLD2

#= This file trains a neural network with the angle of attack and chord values to predict  CL and CD values given by lift_to_drag_alpha.jl and then
lets us compute the value using the neural net for any distribution of chord lengths and gives the true value=#

function get_data(num_inputs, num_outputs,file) # Reads, processes, and splits our data from the vlm file into sets for training and testing
    num_vars = num_outputs + num_inputs
    println("Reading File") # Read data from our vlm file
    lines = readlines(file)
    num_lines = length(lines)
    vlm_data = zeros(num_vars, num_lines)
    for i in 1:num_lines #Split the data up into 8 columns within a matrix
        line = split(lines[i])
        for j in 1:num_vars
            vlm_data[j,i] = parse(Float32,line[j])
        end
    end
    # vlm_data is ordered with cl coefficients, cd coefficients, Alpha, Chord values, and twist values
    # Define test and trainining sets
    println("Defining vectors of data")
    ratio = 0.8 # We want 80 percent of our data set used for training, the rest for testing
    train_size = round(Int, num_lines * ratio)
    test_size = num_lines - train_size
    training_vars = zeros(Float32,num_inputs,train_size) # Create zero lists for training and testing data sets
    training_vlm = zeros(Float32,num_outputs,train_size)
    test_vars = zeros(Float32,num_inputs,test_size)
    test_vlm = zeros(Float32,num_outputs,test_size)
    # Normalize Data
    println("Normalizing the data")
    vlm_norm = zeros(num_vars, 2) # 1-num_vars for each variable within the data, 1-2 for mean and standard deviation respectively
    for i in 1:num_vars # Iterates through each variable and adds mean and standard deviation to vlm_norm matrix
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
        training_vlm[1:num_outputs,i] = vlm_data[1:num_outputs,train_ind[i]] # adds values from selected indices to training set outputs
        for j in num_outputs+1:num_vars
            training_vars[j-num_outputs,i] = vlm_data[j,train_ind[i]] # adds values from selected indices to training set inputs
        end
    end
    for i in 1:num_lines - train_size
        test_vlm[1:num_outputs,i] = vlm_data[1:num_outputs,test_ind[i]] # adds values from selected indices to testing set outputs
        for j in num_outputs+1:num_vars
            test_vars[j-num_outputs,i] = vlm_data[j,test_ind[i]] # adds values from selected indices to testing set inputs
        end
    end
    return training_vars, training_vlm, test_vars, test_vlm, vlm_norm
end

function train(inputs, outputs)
    # Extract data from vlm_alpha_data_file
    training_vars, training_vlm, test_vars, test_vlm, vlm_norm = get_data(inputs, outputs,"vlm_neural_net/vlm_data_file.data")
    ml = 85
    model_vars = (inputs, outputs, ml)
    #region Model Definition
    println("Defining Model")
    model = Lux.Chain(
        Lux.Dense(inputs => ml,relu),
        Lux.Dense(ml => ml,relu),
        Lux.Dense(ml => ml,relu),
        Lux.Dense(ml => ml,relu),
        Lux.Dense(ml => ml,relu),
        Lux.Dense(ml => outputs) 
    )
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    epochs = 1000 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DEFINE EPOCHS HERE
    # __________________________________________________________________________________________________________
    train_losses = ones(epochs) # Create local lists for training and testing losses
    test_losses = ones(epochs)
    cl_losses = ones(epochs)
    cd_losses = ones(epochs)
    #endregion
    #region Loss Functions
    # Define loss function using Mean Squared
    println("Defining loss function")
    cl_loss_weight = 1
    cd_loss_weight = 1
    function loss(model, ps, st, x, y) #= Loss function that deals with both CL and CD, returning losses for each as well as the combined
        loss used for optimization in our model=#
        true_y, new_state = model(x, ps, st)
        lst_length = size(true_y, 1)
        loss_lst = [begin
                    # Get the row for this specific output (c)
                    cfs = @view true_y[c,:] 
                    y_c = @view y[c,:]
                    # Calculate MSE for this row
                    l_cfs = sum(abs2, (cfs .- y_c)) / size(y, 2)
                    l_cfs # Return the loss for this row
                    end for c in 1:lst_length]
        l_cl = loss_lst[1:Int64(outputs/2)]
        l_cd = loss_lst[Int64(outputs/2 +1):end]
        l_CL = sum(l_cl[:])/size(l_cl,1)*cl_loss_weight
        l_CD = sum(l_cd[:])/size(l_cd,1)*cd_loss_weight
        l = l_CL + l_CD
        return l, l_CL, l_CD, new_state
    end
    function abs_loss(model, ps, st, x, y)
        true_y, new_state = model(x, ps, st)
        # Denormalize our values for CL and CD to provide true errors
        true_y2 = zeros(eltype(y),size(true_y))
        y2 = zeros(eltype(y),size(y))
        abs_loss_lst = zeros(Float32, size(true_y, 1))
        lst_length = size(true_y, 1)
        for c in 1:lst_length
            true_y2[c, :] .= true_y[c,:].*vlm_norm[c,2] .+vlm_norm[c,1]
            y2[c,:] .= y[c,:].*vlm_norm[c,2] .+vlm_norm[c,1]
            l_cfs = sum(abs.(true_y2[c,:].-y2[c,:]))/size(y, 2)
            abs_loss_lst[c] = l_cfs
        end
        l_cl = abs_loss_lst[1:Int64(outputs/2)]
        l_cd = abs_loss_lst[Int64(outputs/2 +1):end]
        l_CL = sum(l_cl[:])/size(l_cl,1)
        l_CD = sum(l_cd[:])/size(l_cd,1)
        return l_CL, l_CD, new_state
    end
    #endregion
    #region Optimization
    start_lr = 0.002f0
    min_lr = 0.00001f0
    optimizer = Optimisers.ADAMW(Float32(start_lr), (0.9f0, 0.999f0), 0.05f0) # Using ADAMW as decay helps increase generalization and reduce test loss
    opt_state = Optimisers.setup(optimizer, ps) # Set states
    # Run iterations with epochs
    println("Training in progress...")
    batch_size = 120 # <<<<<<<<<<<<<<<<<<<<<<<<<< BATCH SIZE HERE
    loss_val = 1 
    cl_loss = 1 # create local loss values to update in for loop
    cd_loss = 1 #                                              '
    test_loss_val = 1
    final_cl_error, final_cd_error = 0, 0 # '                   '
    loader = DataLoader((training_vars, training_vlm), batchsize=batch_size, shuffle=true) # Set up batch loading
    local st_loop = st # Manage state inside loop
    global_step_counter = 0 # Step counter for scheduler
    lowest_lr = start_lr
    for epoch in 1:epochs # Iterates through epochs iterations
        local epoch_loss = 0.0 # Defines loss of this epoch iteration
        local num_batches = 0 
        new_lr = 1 # Just do define in the function
        for (x_batch, y_batch) in loader
            global_step_counter += 1
            current_lr = Float32(min(1*test_loss_val/100,start_lr,lowest_lr)) # Define changing learning rate
            if current_lr < lowest_lr
                lowest_lr = current_lr
            end
            new_lr = max(current_lr, min_lr)
            Optimisers.adjust!(opt_state, Float32(new_lr))
            (loss_val, cl_loss, cd_loss, updated_state), grads = Zygote.withgradient(
                p -> loss(model, p, st_loop, x_batch, y_batch), # Use the batch
                ps
            )
            # Avoid runaway gradient
            clipping_threshold = 1.0f0
            grad_norm = LinearAlgebra.norm(grads[1])
            local grads_to_use
            if grad_norm > clipping_threshold
                scaling_factor = clipping_threshold / (grad_norm + 1.0f-6)
                grads_to_use = map(layer -> map(param -> param * scaling_factor, layer), grads[1])
            else
                grads_to_use = grads[1]
            end
            opt_state, ps = Optimisers.update(opt_state, ps, grads_to_use)
            st_loop = updated_state # Update state and other variables
            epoch_loss += loss_val
            num_batches += 1
        end
        current_loss = loss_val
        #train_losses[epoch] = current_loss
        train_losses[epoch] = epoch_loss / num_batches # Used for plotting
        test_loss_val, _ = loss(model, ps, st_loop, test_vars, test_vlm) 
        test_losses[epoch] = test_loss_val # Used for plotting loss, also printed after every 100 epochs
        cl_losses[epoch] = cl_loss # Used for plotting loss
        cd_losses[epoch] = cd_loss # Used for plotting loss
        if epoch == epochs
            final_cl_error, final_cd_error, _ = abs_loss(model, ps, st_loop, test_vars, test_vlm) # Save final absolute errors for cl and cd
        end
        if epoch % 10 == 0
            println("\nEpoch $epoch/$epochs current test loss: $test_loss_val") # Print test loss every 100 epochs
            println("Current lr: $new_lr")
        end
        if epoch % 50 == 0
            loss_plt = plot(1:epoch,train_losses[1:epoch],label="Training Set Loss",title="Loss over time",y_scale=:log10)
            plot!(1:epoch,test_losses[1:epoch],label="Testing Set Loss")
            plot!(1:epoch,cl_losses[1:epoch],label="Lift Loss")
            plot!(1:epoch,cd_losses[1:epoch],label="Drag Loss")
            savefig(loss_plt, "vlm_neural_net/vlm_Loss_Function.png")
        end
    end
    st = st_loop
    #endregion
    #region Loss Plotting
    loss_plt = plot(1:epochs,train_losses,label="Training Set Loss",title="Loss over time",y_scale=:log10)
    plot!(1:epochs,test_losses,label="Testing Set Loss")
    plot!(1:epochs,cl_losses,label="Lift Loss")
    plot!(1:epochs,cd_losses,label="Drag Loss")
    savefig(loss_plt, "vlm_neural_net/vlm_Loss_Function.png")
    #endregion
    #region Plotting CL and CD vs Predicted
    
    # Add some CL and CD values to CL and CD lists for graphing
    data_n = 10 # How many samples of correlating CD and CL along a span. The whole set will be graphed, this is how many sets
    samples = size(test_vars, 2)
    CL_lst = zeros(samples*data_n)
    Pred_CL_lst = zeros(samples*data_n)
    CD_lst = zeros(samples*data_n)
    Pred_CD_lst = zeros(samples*data_n)
    sw = Int64(outputs/2) # Spanwise points, which is half the number of outputs
    for i in 0:data_n-1
        output_lst, _ = Lux.apply(model, test_vars[:, i+1], ps, st)
        for j in 1:Int64(outputs/2)
            CL_lst[i*sw+j] = test_vlm[j,i+1]*vlm_norm[j,2] + vlm_norm[j,1]
            CD_lst[i*sw+j] = test_vlm[j+sw,i+1]*vlm_norm[j+sw,2] + vlm_norm[j+sw,1]
            Pred_CL_lst[i*sw+j] = output_lst[j]*vlm_norm[j,2] + vlm_norm[j,1]
            Pred_CD_lst[i*sw+j] = output_lst[j+sw]*vlm_norm[j+sw,2] + vlm_norm[j+sw,1]
        end
    end
    CL_plt = plot(CL_lst, Pred_CL_lst,label ="", title="Coefficient of Lift",xlabel="Actual", ylabel="Predicted", seriestype=:scatter)
    plot!(CL_plt,CL_lst,CL_lst,label ="")
    savefig(CL_plt, "vlm_neural_net/lift_coefficient.png")
    CD_plt = plot(CD_lst, Pred_CD_lst,label ="", title="Coefficient of Drag",xlabel="Actual", ylabel="Predicted", seriestype=:scatter)
    plot!(CD_plt,CD_lst,CD_lst, label = "")
    savefig(CD_plt, "vlm_neural_net/drag_coefficient.png")
    # Plotting a set of coefficients against the actual for lift and drag along the span
    Cl_span = plot(range(-1,1, length=sw),CL_lst[1:sw],label="Actual From VLM",xlabel="Span",ylabel="Lift Coefficients")
    plot!(Cl_span,range(-1,1,length=sw),Pred_CL_lst[1:sw], label="Predicted From Model")
    savefig(Cl_span, "vlm_neural_net/lift_coefficient_span.png")
    Cd_span = plot(range(-1,1, length=sw),CD_lst[1:sw],label="Actual From VLM",xlabel="Span",ylabel="Drag Coefficients")
    plot!(Cd_span,range(-1,1,length=sw),Pred_CD_lst[1:sw], label="Predicted From Model")
    savefig(Cd_span, "vlm_neural_net/drag_coefficient_span.png")

    #endregion

    return model_vars, ps, st, final_cl_error, final_cd_error, vlm_norm
end

function main()
    inputs = 23 # These are our inputs including alpha, chord, and twist distributions
    outputs = 40 # <<<<<<<<<<<<<<<<<<< MUST EDIT IF N_OUTPUTS CHANGES  This is four times the spanwise spacing used to set up the VLM analysis
    model_vars, ps, st, cl_error, cd_error, vlm_norm = @time train(inputs, outputs) # run train function to train the neural network
    println("Test Set CL Average Absolute Error: $cl_error")
    println("Test Set CD Average Absolute Error: $cd_error")
    println("Would you like to save the model? (Y/N)")
    answer = readline()
    if answer == "Y" || answer == "y"
        println("What name would you like to save the model as?")
        name = readline()
        println("Saving model to 'vlm_neural_net/models/$name.jld2'...")
        save("vlm_neural_net/models/$name.jld2", Dict("ps" => ps, "st" => st, "vlm_norm" => vlm_norm, "model_vars" => model_vars))
    end
end

