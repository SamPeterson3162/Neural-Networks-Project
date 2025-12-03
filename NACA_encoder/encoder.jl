using Lux
using Plots
using Optimisers
using Zygote
using Random
using Statistics
using MLUtils
using LinearAlgebra
using JLD2
using DelimitedFiles


function NACA_numbers(n)

    nacas = fill("", n)
    for i in 1:n
        #Create random numbers that are realistic for NACA
        m = rand(0:9)
        p = rand(0:9)
        t = rand(5:30)
        code = "$(m)$(p)$(string(t, pad=2))"
        nacas[i] = code
    end
    return nacas
end

function create_coordinates(code::String, n_points::Int=100)
    #Parse the 4-digit NACA number
    m = parse(Int, code[1:1]) / 100.0 #Maximum camber of the airfoil
    p = parse(Int, code[2:2]) / 10.0 #Position of maximum camber
    t = parse(Int, code[3:4]) / 100.0 #Maximum thickness as percentage of 

    #Generate x values (cosine spacing for smoother leading edge with more points)
    beta = range(0, π, length=n_points)
    x = (1.0 .- cos.(beta)) ./ 2.0  # Points clustered at LE and TE

    #Calculate Thickness (yt)
    #Coefficients for NACA 4-digit series
    a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
    
    yt = @. 5 * t * (a0 * sqrt(x) + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4)

    #Calculate Camber (yc) and Gradient (dy/dx)
    yc = zeros(n_points)
    dy_dx = zeros(n_points)

    if m == 0  # Symmetric airfoil (e.g., 0012)
        # Camber is 0, Gradient is 0
        fill!(yc, 0.0)
        fill!(dy_dx, 0.0)
    else
        for i in 1:n_points
            if x[i] < p
                yc[i] = (m / p^2) * (2 * p * x[i] - x[i]^2)
                dy_dx[i] = (2 * m / p^2) * (p - x[i])
            else
                yc[i] = (m / (1 - p)^2) * ((1 - 2 * p) + 2 * p * x[i] - x[i]^2)
                dy_dx[i] = (2 * m / (1 - p)^2) * (p - x[i])
            end
        end
    end

    #Calculate Slope Angle (theta)
    theta = atan.(dy_dx)

    #Calculate Upper and Lower Surface Coordinates
    # (Trailing Edge -> Leading Edge -> Trailing Edge)
    xu = @. x - yt * sin(theta)
    yu = @. yc + yt * cos(theta)
    
    xl = @. x + yt * sin(theta)
    yl = @. yc - yt * cos(theta)

    # Combine into a single list of (x, y) coordinates
    # Upper surface (TE to LE) -> Lower surface (LE to TE)
    # We drop the first point of lower to avoid duplicate LE point
    
    final_x = vcat(reverse(xu), xl[2:end])
    final_y = vcat(reverse(yu), yl[2:end])

    return final_x, final_y
end

function get_naca_data(nacas, n_coords)
    n = length(nacas)
    tot_coords = (n_coords*4)-2
    naca_data = zeros(tot_coords, n)
    for i in 1:n
        x_data, y_data = create_coordinates(nacas[i],n_coords)
        naca_data[1:n_coords*2-1,i] .= x_data
        naca_data[n_coords*2:end,i] .= y_data
    end
    writedlm("NACA_encoder/naca_coordinates.data", transpose(naca_data), " ")
    return naca_data
end

function plt()
    # Plots 1 random NACA airfoil
    n_naca = 1 # We only need one NACA as this is just checking geometry
    n_coords = 50

    nacas = NACA_numbers(n_naca) # Gives n different random NACA airfoils
    naca_data = get_naca_data(nacas,n_coords)
    # --- Usage Example ---
    x_coords, y_coords = naca_data[1:n_coords*2-1,1], naca_data[n_coords*2:end,1]

    # Visualize it
    plot(x_coords, y_coords, aspect_ratio=:equal, title=nacas[1], label = "")
end

function train_neural_network(n_naca,n_coords,n_digits,naca_data)
    n_inputs = n_coords*4-2
    #region Preprocessing
    println("Normalizing Airfoil Data...")
    input_matrix = zeros(size(naca_data))
    stats_matrix = zeros(n_inputs, 2)
    for i in 1:n_inputs
        stats_matrix[i,1] = mean(naca_data[i,:])
        stats_matrix[i,2] = std(naca_data[i,:])
        if stats_matrix[i,2] == 0
            input_matrix[i,:] .= 0
        else
            input_matrix[i,:] .= (naca_data[i,:] .- stats_matrix[i,1]) ./ stats_matrix[i,2]
        end
    end
    println("Generating Training and Testing Sets...")
    split_index = round(Int, n_naca*0.8) # Split the training and testing sets 80/20
    train_vec = input_matrix[:,1:split_index] # Normalized training vector for direct input 
    test_vec = input_matrix[:,split_index+1:end] # Normalized testing vector
    comparison_vec = naca_data[:,split_index+1:end] # Unnormalized for use in absolute loss
    #endregion
    #region Model Definition
    # Generate a Model using two parts: An encoder and a decoder, with n_digits in the middle
    central_neurons = 100
    model_vars = (n_inputs, central_neurons, n_digits)
    encoder = Lux.Chain(
        Lux.Dense(n_inputs => central_neurons, relu),
        Lux.Dense(central_neurons => n_digits)
    )
    decoder = Lux.Chain(
        Lux.Dense(n_digits => central_neurons, relu),
        Lux.Dense(central_neurons => n_inputs)
    )
    model = Lux.Chain(encoder = encoder, decoder = decoder) 
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    epochs = 5000 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DEFINE EPOCHS HERE
    train_losses = ones(epochs) # Create local lists for training and testing losses
    test_losses = ones(epochs)
    x_losses = ones(epochs)
    y_losses = ones(epochs)
    # Define Loss Functions
    function loss(model, ps, st, x, y) #Loss function that will use MSE for all the inputs to outputs
        true_y, new_state = model(x, ps, st)
        lst_length = size(true_y, 1)
        loss_lst = [begin
                    # Get the row for this specific output
                    coords = @view true_y[c,:] 
                    y_c = @view y[c,:]
                    # Calculate MSE for this row
                    l_coords = sum(abs2,(coords .- y_c)) / size(y, 2)
                    l_coords # Return the loss for this row
                    end for c in 1:lst_length]
        l_x = loss_lst[1:Int64(n_inputs/2)]
        l_y = loss_lst[Int64(n_inputs/2 +1):end]
        l_X = sum(l_x[:])/size(l_x,1)
        l_Y = sum(l_y[:])/size(l_y,1)
        total_loss = (l_X+l_Y)/2
        return total_loss, l_X, l_Y, new_state
    end
    function absolute_loss(model, ps, st, x, y) #Loss function that compares unnormalized data to outputs of autoencoder
        true_y, new_state = model(x, ps, st)
        pred_unnormalized = (true_y .* stats_matrix[:, 2]) .+ stats_matrix[:, 1]
        diff = pred_unnormalized .- y # y is comparison_vec (already unnormalized)
        half_idx = Int(size(diff, 1) / 2)
        x_error_matrix = @view diff[1:half_idx, :]
        l_X = mean(abs.(x_error_matrix)) 
        y_error_matrix = @view diff[half_idx+1:end, :]
        l_Y = mean(abs.(y_error_matrix))
        return l_X, l_Y, new_state
    end
    #endregion
    #region Optimization
    start_lr = 0.002f0
    min_lr = 0.00001f0
    optimizer = Optimisers.OptimiserChain(
        Optimisers.ClipGrad(1.0f0),
        Optimisers.ADAMW(Float32(start_lr), (0.9f0, 0.999f0), 0.05f0) # Using ADAMW as decay helps increase generalization and reduce test loss
    )
    opt_state = Optimisers.setup(optimizer, ps) # Set states
    # Run iterations with epochs
    println("Training in progress...")
    batch_size = 64 # <<<<<<<<<<<<<<<<<<<<<<<<<< BATCH SIZE HERE
    loss_val = 1 
    x_loss = 1 # create local loss values to update in for loop
    y_loss = 1 #                                              '
    test_loss_val = 1
    final_x_error, final_y_error = 0, 0 # '                   '
    loader = DataLoader((train_vec, train_vec), batchsize=batch_size, shuffle=true) # Set up batch loading
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
            (loss_val, x_loss, y_loss, updated_state), grads = Zygote.withgradient(
                p -> loss(model, p, st_loop, x_batch, y_batch), # Use the batch
                ps
            )
            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
            st_loop = updated_state # Update state and other variables
            epoch_loss += loss_val
            num_batches += 1
        end
        current_loss = loss_val
        #train_losses[epoch] = current_loss
        train_losses[epoch] = epoch_loss / num_batches # Used for plotting
        test_loss_val, _ = loss(model, ps, st_loop, test_vec, test_vec) 
        test_losses[epoch] = test_loss_val # Used for plotting loss, also printed after every 100 epochs
        x_losses[epoch] = x_loss # Used for plotting loss
        y_losses[epoch] = y_loss # Used for plotting loss
        if epoch == epochs
            final_x_error, final_y_error, _ = absolute_loss(model, ps, st_loop, test_vec, comparison_vec) # Save final absolute errors for cl and cd
        end
        if epoch % 10 == 0
            println("\nEpoch $epoch/$epochs current test loss: $test_loss_val") # Print test loss every 100 epochs
            println("Current lr: $new_lr")
        end
        if epoch % 50 == 0
            loss_plt = plot(1:epoch,train_losses[1:epoch],label="Training Set Loss",title="Loss over time",y_scale=:log10)
            plot!(1:epoch,test_losses[1:epoch],label="Testing Set Loss")
            plot!(1:epoch,x_losses[1:epoch],label="X Loss")
            plot!(1:epoch,y_losses[1:epoch],label="Y Loss")
            savefig(loss_plt, "NACA_encoder/figures/encoder_Loss_Function.png")
        end
    end
    println()
    println("Final x-coordinate error: $final_x_error")
    println("Final y-coordinate error: $final_y_error")
    st = st_loop
    #endregion
    #region Loss Plotting
    loss_plt = plot(1:epochs,train_losses,label="Training Set Loss",title="Loss over time",y_scale=:log10)
    plot!(1:epochs,test_losses,label="Testing Set Loss")
    plot!(1:epochs,x_losses,label="X Loss")
    plot!(1:epochs,y_losses,label="Y Loss")
    savefig(loss_plt, "NACA_encoder/figures/encoder_Loss_Function.png")
    #endregion
    println("Would you like to save the model? (Y/N)")
    answer = readline()
    if answer == "Y" || answer == "y"
        println("What name would you like to save the model as?")
        name = readline()
        println("Saving model to 'vlm_neural_net/models/$name.jld2'...")
        save("NACA_encoder/models/$name.jld2", Dict("ps" => ps, "st" => st, "stats_matrix" => stats_matrix, "model_vars" => model_vars))
    end
    return model, ps, st, stats_matrix
end

function visualize_code(model, ps, st, nacas, n_naca, n_coords, naca_data, stats_matrix)
    n_inputs = n_coords*4-2
    println("Normalizing Visualization Data...")
    input_matrix = zeros(size(naca_data))
    for i in 1:n_inputs
        if stats_matrix[i,2] == 0
            input_matrix[i,:] .= 0
        else
            input_matrix[i,:] .= (naca_data[i,:] .- stats_matrix[i,1]) ./ stats_matrix[i,2]
        end
    end
    m_lst = zeros(n_naca)
    p_lst = zeros(n_naca)
    t_lst = zeros(n_naca)
    for i in 1:n_naca
        m_lst[i] = parse(Int, nacas[i][1:1]) / 100.0 
        p_lst[i] = parse(Int, nacas[i][2:2]) / 10.0 
        t_lst[i] = parse(Int, nacas[i][3:4]) / 100.0 
    end
    encoder_model = model.layers.encoder
    latent_vector, _ = Lux.apply(encoder_model, input_matrix, ps.encoder, st.encoder)
    # Latent coordinates (x, y, z)
    lx = latent_vector[1,:]
    ly = latent_vector[2,:]
    lz = latent_vector[3,:]
    # Use marker_z to color the points by their physical property
    p1 = scatter(lx, ly, lz, marker_z=m_lst, color=:plasma, 
                 title="Colored by Max Camber (M)", label="", 
                 xlabel="Lat 1", ylabel="Lat 2", zlabel="Lat 3",
                 camera=(30, 30), markerstrokewidth=0, markersize=3, alpha=0.8)

    p2 = scatter(lx, ly, lz, marker_z=p_lst, color=:viridis, 
                 title="Colored by Camber Pos (P)", label="", 
                 xlabel="Lat 1", ylabel="Lat 2", zlabel="Lat 3",
                 camera=(30, 30), markerstrokewidth=0, markersize=3, alpha=0.8)

    p3 = scatter(lx, ly, lz, marker_z=t_lst, color=:turbo, 
                 title="Colored by Thickness (T)", label="", 
                 xlabel="Lat 1", ylabel="Lat 2", zlabel="Lat 3",
                 camera=(30, 30), markerstrokewidth=0, markersize=3, alpha=0.8)
    # Display them all together
    savefig(p1, "NACA_encoder/figures/plotted_max_camber_correlation.png")
    savefig(p2, "NACA_encoder/figures/plotted_camber_positon_correlation.png")
    savefig(p3, "NACA_encoder/figures/plotted_thickness_correlation.png")
    # This prints how much each neuron cares about M, P, or T
    println("\n--- Correlation Analysis ---")
    println("Rows = Latent Neurons (1, 2, 3)")
    println("Cols = Physics Parameters (M, P, T)")
    
    # Stack latent vectors and physics vectors
    latent_matrix = Matrix(latent_vector') # Transpose to (Samples x 3)
    physics_matrix = hcat(m_lst, p_lst, t_lst) # (Samples x 3)
    
    # Calculate correlation
    cor_mat = cor(latent_matrix, physics_matrix)
    display(round.(cor_mat, digits=2))
end

function predict_naca_data(naca_data, model, ps, st, stats_matrix, n_coords)
    model_data = zeros(size(naca_data))
    predicted_data = zeros(size(naca_data))
    coords = size(naca_data,1)
    for i in 1:coords
        if stats_matrix[i,2] == 0
            model_data[i,1] = 0
        else
            model_data[i,:] .= (naca_data[i,:] .- stats_matrix[i,1]) ./ stats_matrix[i,2]
        end
    end
    predicted_data, _ = Lux.apply(model, model_data, ps, st)
    for i in 1:coords
        predicted_data[i,:] .= (predicted_data[i,:] .* stats_matrix[i,2]) .+ stats_matrix[i,1]
    end
    return predicted_data
end

function build_model(model_vars)
    n_inputs, central_neurons, n_digits = model_vars
    encoder = Lux.Chain(
        Lux.Dense(n_inputs => central_neurons, relu),
        Lux.Dense(central_neurons => n_digits)
    )
    decoder = Lux.Chain(
        Lux.Dense(n_digits => central_neurons, relu),
        Lux.Dense(central_neurons => n_inputs)
    )
    return Lux.Chain(encoder = encoder, decoder = decoder)
end

function load_neural_network(filename)
    data = load(filename) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MODEL 
    ps_loaded = data["ps"]
    st_loaded = data["st"]
    stats_matrix = data["stats_matrix"]
    model_vars = data["model_vars"]
    return build_model(model_vars), ps_loaded, st_loaded, stats_matrix
end

function foil()
    n_coords = 25 # Run the same as the trained model
    nacas = NACA_numbers(5)
    naca_data = get_naca_data(nacas,n_coords)
    model, ps, st, stats_matrix = load_neural_network("NACA_encoder/models/5000_epochs.jld2")
    naca_data2 = predict_naca_data(naca_data, model, ps, st, stats_matrix, n_coords)
    points = n_coords*2-1
    for i in 1:5
        naca_data_x = naca_data[1:points, i]
        naca_data_y = naca_data[points+1:end, i]
        naca_data2_x = naca_data2[1:points, i]
        naca_data2_y = naca_data2[points+1:end, i]
        plt = plot(naca_data_x, naca_data_y, title="Airfoil Comparison", label="Actual", aspect_ratio=:equal)
        plot!(naca_data2_x, naca_data2_y,label="Decoded")
        savefig(plt, "NACA_encoder/figures/airfoil_sample_$i.png")
    end
end

function main(command::String="")
    n_naca = 2000 # How many different NACA foils we will train with
    n_coords = 25
    n_digits = 3 # Digits we would like to encode down to
    nacas = NACA_numbers(n_naca) # Gives n different random NACA airfoils
    naca_data = get_naca_data(nacas,n_coords)
    if command == "train"
        model, ps, st, stats_matrix = train_neural_network(n_naca, n_coords, n_digits, naca_data)
    else
        model, ps, st, stats_matrix = load_neural_network("NACA_encoder/models/5000_epochs.jld2")
    end
    visualize_code(model, ps, st, nacas, n_naca, n_coords, naca_data, stats_matrix)
end
