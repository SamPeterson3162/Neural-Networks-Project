# Load necessary Packages
using Lux
using Plots
using Optimisers
using Zygote
using Random

#= We will make a 1D function, create training data with noise, use it to train our model, and then plot the date with
the actual function side by side to show the loss. We will also crate a line for input to apply the trained model=#

function train(true_func)
    # Create x and y training functions
    x_start = -10
    x_end = 10
    num_samples = 100
    x_train = collect(range(x_start, x_end, length=num_samples))
    y_train = true_func.(x_train)
    x_train = reshape(x_train,1, num_samples)
    y_train = reshape(y_train,1, num_samples)
    # Define our model
    model = Lux.Chain(
        Lux.Dense(1 => 50,relu),
        Lux.Dense(50 => 50,relu),
        Lux.Dense(50 => 50,relu),
        Lux.Dense(50 => 1)
    )
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    # Set up the optimization
    max_iterations = 4000
    current_iteration = 0
    # Iterations will stop when the target loss is achieved.
    target_loss = .05
    losses = ones(max_iterations).*target_loss
    current_loss = 1
    # Define loss function using mean Squared
    function loss(model, ps, st, x, y)
            true_y, new_state = model(x, ps, st)
            l = sum((true_y.-y).^2)/num_samples
            return l, new_state
        end
    optimizer = Optimisers.ADAM(0.0005f0)
    opt_state = Optimisers.setup(optimizer, ps)
    while current_iteration < max_iterations && current_loss > target_loss # Iterates until either target loss or max iterations are met
        (loss_val, updated_state), grads = Zygote.withgradient(
            p -> loss(model, p, st, x_train, y_train),
            ps
        )
        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        st = updated_state
        current_loss = loss_val
        current_iteration += 1
        losses[current_iteration] = current_loss
        if current_iteration % 100 == 0
            println("\ncurrent loss: $current_loss")
        end
    end
    println("Final Loss Acheived: $current_loss")
    losses = losses[1:current_iteration]
    loss_range = collect(range(1, current_iteration))
    #region Plotting
    # Plot training points with the true function and the predicted values from the model
    x_plot = reshape(collect(range(-10.0f0, 10.0f0, length=100)), 1, 100)
    y_pred, _ = Lux.apply(model, x_plot, ps, st)
    plot(vec(x_train), vec(y_train), seriestype=:scatter, label="Training Data",
        title="NN Learning Linear Function", color=:white)
    plot!(vec(x_plot), vec(y_pred), label="NN Prediction (Learned)", linewidth=3, linestyle=:dash, color=:blue)
    plot!(vec(x_plot), true_func.(vec(x_plot)), label="True Function", linewidth=1, color=:red)
    xlabel!("x")
    ylabel!("y")
    savefig("nn_linear_regression_plot.png")
    # Plot the loss over time
    println("Plot saved as nn_linear_regression_plot.png")
    plot(loss_range, losses, title="Loss Over Time",label = "",y_scale=:log10)
    savefig("loss_over_time.png")
    # Plot outer range to show how the model is inaccurate outside the range.
    x_plot = reshape(collect(range(-20.0f0, 20.0f0, length=100)), 1, 100)
    plot(vec(x_plot), true_func.(vec(x_plot)), label="True Function")
    y_pred, _ = Lux.apply(model, x_plot, ps, st)
    plot!(vec(x_plot), vec(y_pred), label="NN Prediction (Learned)", linewidth=3, linestyle=:dash, color=:blue)
    xlabel!("x")
    ylabel!("y")
    savefig("outside_range.png")
    #endregion


    return model, ps, st
end

function main()
    true_func(x) = x^2 + 2*x - 4
    model, ps, st = train(true_func)
    prediction(x) = Lux.apply(model, x, ps, st)
    while true # While loop where we can compute any value with the trained model
        println("What x-value would you like me to compute the y-value for?")
        x_val = parse(Float32,readline())
        x_val_1 = [x_val]
        y_val, _ = prediction(x_val_1)
        true_y = true_func(x_val)
        y_val = y_val[1]
        println("X-Value:           $x_val")
        println("Predicted Y-Value: $y_val")
        println("True Y-Value:      $true_y")
        println("Would you like me to compute another value? (\"N\" to quit)")
        ans = readline()
        if (ans == "N"|| ans == "n")
            break
        end
    end
end
