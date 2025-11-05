# Load necessary Packages
using Lux
using Plots
using Optimisers
using Zygote
using Random
using Statistics
using MLUtils

function get_data(file)
    println("Reading File")
    lines = readlines(file)
    num_lines = length(lines)
    mpg_data = zeros(8, num_lines)
    for i in 1:num_lines
        line = split(lines[i])
        for j in 1:8
            mpg_data[j,i] = parse(Float32,line[j])
        end
    end
    # Define test and trainining sets
    println("Defining vectors of data")
    ratio = 0.8
    train_size = round(Int, num_lines * ratio)
    test_size = num_lines - train_size
    training_vars = zeros(7,train_size)
    training_mpg = zeros(1,train_size)
    test_vars = zeros(7,test_size)
    test_mpg = zeros(1,test_size)
    # Normalize Data
    println("Normalizing the data")
    mpg_norm = [0.0,0.0]
    for i in 1:8
        c_mean = mean(mpg_data[i,:])
        c_std = std(mpg_data[i,:])
        mpg_data[i,:] .= (mpg_data[i,:].-c_mean)./c_std
        # Save mean and standard deviation for mpg data set
        if i == 1
            mpg_norm[1] = c_mean
            mpg_norm[2] = c_std
        end
    end
    # Split mpg_data using the 80/20 split for training and testing
    println("Defining training and testing sets")
    indices = shuffle!(collect(1:num_lines))
    train_ind = indices[1:train_size]
    test_ind = indices[train_size+1:end]
    for i in 1:train_size
        training_mpg[1,i] = mpg_data[1,train_ind[i]]
        for j in 2:8
            training_vars[j-1,i] = mpg_data[j,train_ind[i]]
        end
    end
    for i in 1:num_lines - train_size
        test_mpg[1,i] = mpg_data[1,test_ind[i]]
        for j in 2:8
            test_vars[j-1,i] = mpg_data[j,test_ind[i]]
        end
    end
    return training_vars, training_mpg, test_vars, test_mpg, mpg_norm
end

function train()
    # Extract data from auto-mpg.data file
    training_vars, training_mpg, test_vars, test_mpg, mpg_norm = get_data("auto+mpg/auto-mpg.data")
    # Define our model
    println("Defining Model")
    num_samples = size(training_vars,2)
     model = Lux.Chain(
        Lux.Dense(7 => 24,relu),
        Lux.Dense(24 => 10,relu),
        Lux.Dense(10 => 1)
    )
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    epochs = 2500
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
        true_y2 .= true_y.*mpg_norm[2] .+mpg_norm[1]
        y2 .= y.*mpg_norm[2] .+mpg_norm[1]
        l = sum(abs.(true_y2.-y2))/size(y, 2)

        return l, new_state
    end
    optimizer = Optimisers.ADAMW(0.0002f0, (0.9, 0.999), .05f0)
    opt_state = Optimisers.setup(optimizer, ps)
    # Run iterations with epochs
    println("Training in progress...")
    batch_size = 64
    loss_val = 1
    loader = DataLoader((training_vars, training_mpg), batchsize=batch_size, shuffle=true)
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
        test_loss_val, _ = loss(model, ps, st, test_vars, test_mpg)
        test_losses[epoch] = test_loss_val
        if epoch == epochs
            final_error, _ = abs_loss(model, ps, st, test_vars, test_mpg)
        end
        if epoch % 100 == 0
            println("\ncurrent loss: $current_loss")
        end
    end
    loss_plt = plot(1:epochs,train_losses,label="Training Set Loss",title="Loss over time",y_scale=:log10)
    plot!(1:epochs,test_losses,label="Testing Set Loss")
    savefig(loss_plt, "MPG_Loss_Function.png")
    final_vars = test_vars[:,1]
    final_mpg = test_mpg[1,1]

    return model, ps, st, mpg_norm, final_vars, final_mpg, final_error
end

function main()
    model, ps, st, mpg_norm, final_vars, final_mpg, final_error = train()
    println("Test Set Average Absolute Error: $final_error MPG")
    # Apply the model to a random line of data
    prediction(x) = Lux.apply(model, x, ps, st)
    test_vals = final_vars
    n_mpg_val, _ = prediction(test_vals)
    mpg_val = n_mpg_val.*mpg_norm[2] .+ mpg_norm[1]
    mpg_val = mpg_val[1]
    println("Predicted MPG Value: $mpg_val")
    final_mpg = final_mpg.*mpg_norm[2] .+ mpg_norm[1]
    println("Actual MPG Value: $final_mpg")
end

