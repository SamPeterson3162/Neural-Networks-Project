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
using DataInterpolations


function NACA_numbers(n)

    nacas = fill("", n)
    for i in 1:n
        #Create random numbers that are realistic for NACA
        l = rand(1:8)
        p = rand(1:5)
        q = rand(0:1)
        t = rand(5:40)
        code = "$(l)$(p)$(q)$(string(t, pad=2))"
        nacas[i] = code
    end
    return nacas
end


function create_coordinates(code::String, n_points::Int=50)
    # Generate Dense Raw Geometry ---
    n_dense = n_points
    l = parse(Int, code[1:1])*.15
    p = parse(Int, code[2:2]) / 20.0
    q = parse(Int, code[3:3])
    t = parse(Int, code[4:5]) / 100.0
    beta_dense = range(0, π, length=n_dense)
    x = (1.0 .- cos.(beta_dense)) ./ 2.0
    a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
    yt = @. 5 * t * (a0 * sqrt(x) + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4)
    yc = zeros(n_dense)
    dy_dx = zeros(n_dense)
    function get_m_p(l, p, q)
        # Common Position Inputs
        positions = [.05, .10, .15, .20, .25]
        
        if q == 0 # Standard Camber
            m_vals = [0.0580, 0.1260, 0.2025, 0.2900, 0.3910]
            k1_vals = [361.4, 51.64, 15.957, 6.643, 3.230]
            # Standard line has no k2 or R, so return 0 for R
            R_vals = [0.0, 0.0, 0.0, 0.0, 0.0] 
            
        elseif q == 1 # Reflexed Camber
            m_vals = [0.1300, 0.2170, 0.3184, 0.4412, 0.5896]
            k1_vals = [51.99, 15.793, 6.520, 3.191, 1.762]
            # R = k2/k1 ratio
            R_vals = [0.0007, 0.0085, 0.0303, 0.0719, 0.1382]
        end

        # Perform Interpolation
        inter_m = AkimaInterpolation(m_vals, positions)
        inter_k1 = AkimaInterpolation(k1_vals, positions)
        inter_R = AkimaInterpolation(R_vals, positions)

        # Calculate final values
        m = inter_m(p)
        # Scale k1 based on the design lift (l)
        # Note: The table values are for Cl=0.3
        k1 = inter_k1(p) * (l / 0.3)
        R = inter_R(p) # Only relevant if q=1

        return m, k1, R
    end
    m, k1, R = get_m_p(l, p, q)
    if m == 0
        fill!(yc, 0.0); fill!(dy_dx, 0.0)
    else
        if q == 0
            for i in 1:n_dense
                if x[i] < p
                    yc[i] = (k1/6)*(x[i]^3-3*m*x[i]^2+(m^2)*(3-m)*x[i])
                    dy_dx[i] = (k1/6)*(3*x[i]^2-6*m*x[i]+(m^2)*(3-m))
                else
                    yc[i] = ((k1*m^3)/6)*(1-x[i])
                    dy_dx[i] = ((k1*m^3)/6)*(-1)
                end
            end
        else
            for i in 1:n_dense
                if x[i] < p
                    yc[i] = (k1/6)*((x[i]-m)^3-x[i]*R*(1-m)^3-(m^3)*x[i]+m^3)
                    dy_dx[i] = (k1/6)*(3*(x[i]-m)^2-R*(1-m)^3-m^3)
                else
                    yc[i] = (k1/6)*(R*(x[i]-m)^3-x[i]*R*(1-m)^3-(m^3)*x[i]+m^3)
                    dy_dx[i] = (k1/6)*(3*R*(x[i]-m)^2-R*(1-m)^3-m^3)
                end
            end
        end
    end
    theta = atan.(dy_dx)
    # Raw coordinates
    xu_raw = @. x - yt * sin(theta)
    yu_raw = @. yc + yt * cos(theta)
    xl_raw = @. x + yt * sin(theta)
    yl_raw = @. yc - yt * cos(theta)

    # Upper Surface: TE -> LE (reverse xu)
    # Lower Surface: LE -> TE (xl)
    final_x = vcat(reverse(xu_raw), xl_raw[2:end])
    final_y = vcat(reverse(yu_raw), yl_raw[2:end])
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

function train_neural_network(n_naca,n_coords,n_digits,naca_data, epochs)
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
    central_neurons = 80
    model_vars = (n_inputs, central_neurons, n_digits)
    encoder = Lux.Chain(
        Lux.Dense(n_inputs => central_neurons, relu),
        Lux.Dense(central_neurons => central_neurons, relu),
        Lux.Dense(central_neurons => n_digits)
    )
    decoder = Lux.Chain(
        Lux.Dense(n_digits => central_neurons, relu),
        Lux.Dense(central_neurons => central_neurons, relu),
        Lux.Dense(central_neurons => n_inputs)
    )
    model = Lux.Chain(encoder = encoder, decoder = decoder) 
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
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
    batch_size = 100 # <<<<<<<<<<<<<<<<<<<<<<<<<< BATCH SIZE HERE
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

function process_general_airfoil(filename::String, n_points::Int=50)
    # --- 1. Read Sparse File ---
    x_raw, y_raw = Float64[], Float64[]
    for line in eachline(filename)
        parts = split(strip(line))
        if length(parts) >= 2
            try push!(x_raw, parse(Float64, parts[1])); push!(y_raw, parse(Float64, parts[2])) catch end
        end
    end
    if isempty(x_raw); error("Empty file: $filename"); end

    # --- 2. GEOMETRIC ALIGNMENT (De-Rotation) ---
    # A. Find Nose and Tail
    min_val, nose_idx = findmin(x_raw)
    x_nose, y_nose = x_raw[nose_idx], y_raw[nose_idx]
    
    # Tail = Average of first and last points
    x_tail = (x_raw[1] + x_raw[end]) / 2.0
    y_tail = (y_raw[1] + y_raw[end]) / 2.0
    
    # B. Calculate Rotation Angle
    dx = x_tail - x_nose
    dy = y_tail - y_nose
    angle = atan(dy, dx)
    c, s = cos(-angle), sin(-angle)
    
    # C. Rotate
    x_rot = zeros(length(x_raw))
    y_rot = zeros(length(y_raw))
    
    for i in 1:length(x_raw)
        xt = x_raw[i] - x_nose
        yt = y_raw[i] - y_nose
        x_rot[i] = xt * c - yt * s
        y_rot[i] = xt * s + yt * c
    end
    
    # D. Normalize
    chord_len = maximum(x_rot)
    x_norm = x_rot ./ chord_len
    y_norm = y_rot ./ chord_len

    # --- 3. SEPARATE SURFACES ---
    # Re-find nose in normalized coordinates
    _, new_nose_idx = findmin(x_norm)
    
    limb_a_x = x_norm[1:new_nose_idx]
    limb_a_y = y_norm[1:new_nose_idx]
    limb_b_x = x_norm[new_nose_idx:end]
    limb_b_y = y_norm[new_nose_idx:end]
    
    if mean(limb_a_y) > mean(limb_b_y)
        upper_x, upper_y = limb_a_x, limb_a_y
        lower_x, lower_y = limb_b_x, limb_b_y
    else
        upper_x, upper_y = limb_b_x, limb_b_y
        lower_x, lower_y = limb_a_x, limb_a_y
    end

    # --- 4. SQRT TRANSFORM & ANCHORING ---
    u_upper = sqrt.(clamp.(upper_x, 0.0, 1.0))
    u_lower = sqrt.(clamp.(lower_x, 0.0, 1.0))
    
    # Sort
    perm_u = sortperm(u_upper); u_upper = u_upper[perm_u]; y_upper = upper_y[perm_u]
    perm_l = sortperm(u_lower); u_lower = u_lower[perm_l]; y_lower = lower_y[perm_l]

    # Force Anchors (Prevents Spline Extrapolation Errors)
    function ensure_boundary!(u_vec, y_vec, target_u)
        tol = 1e-6
        if abs(u_vec[1] - target_u) > tol && target_u == 0.0
             pushfirst!(u_vec, 0.0); pushfirst!(y_vec, 0.0)
        end
        if abs(u_vec[end] - target_u) > tol && target_u == 1.0
             push!(u_vec, 1.0); push!(y_vec, 0.0)
        end
    end
    
    ensure_boundary!(u_upper, y_upper, 0.0); ensure_boundary!(u_upper, y_upper, 1.0)
    ensure_boundary!(u_lower, y_lower, 0.0); ensure_boundary!(u_lower, y_lower, 1.0)

    # Clean Duplicates
    unique_u = unique(i -> u_upper[i], 1:length(u_upper))
    u_upper = u_upper[unique_u]; y_upper = y_upper[unique_u]
    unique_l = unique(i -> u_lower[i], 1:length(u_lower))
    u_lower = u_lower[unique_l]; y_lower = y_lower[unique_l]

    # --- 5. INITIAL SPLINE (Dense Proxy) ---
    itp_u = AkimaInterpolation(y_upper, u_upper)
    itp_l = AkimaInterpolation(y_lower, u_lower)
    
    # Generate Dense Cloud
    n_dense = 2000
    beta_dense = range(0, π, length=n_dense)
    x_dense_t = (1.0 .- cos.(beta_dense)) ./ 2.0
    u_dense_t = sqrt.(x_dense_t) # 0.0 -> 1.0
    
    y_dense_u = itp_u.(u_dense_t)
    y_dense_l = itp_l.(u_dense_t)

    # --- 6. HIGH-RES LINEAR RESAMPLING (FIXED) ---
    # We pass x_dense_t directly (0 -> 1) because searchsortedlast requires ascending order.
    
    x_upper_dense = x_dense_t # Already 0->1
    y_upper_dense = y_dense_u # Nose->Tail
    
    x_lower_dense = x_dense_t # Already 0->1
    y_lower_dense = y_dense_l # Nose->Tail

    # Target Grid (Cosine)
    beta_target = range(0, π, length=n_points)
    x_target = (1.0 .- cos.(beta_target)) ./ 2.0
    
    function dense_interp(x_t, x_src, y_src)
        i = searchsortedlast(x_src, x_t)
        if i == 0; return y_src[1]; end
        if i >= length(x_src); return y_src[end]; end
        x1, x2 = x_src[i], x_src[i+1]
        y1, y2 = y_src[i], y_src[i+1]
        return y1 + ((x_t - x1) / (x2 - x1)) * (y2 - y1)
    end
    
    y_target_upper = [dense_interp(xt, x_upper_dense, y_upper_dense) for xt in x_target]
    y_target_lower = [dense_interp(xt, x_lower_dense, y_lower_dense) for xt in x_target]

    # --- 7. FORMAT OUTPUT ---
    final_xu = reverse(x_target)
    final_yu = reverse(y_target_upper)
    final_xl = x_target[2:end]
    final_yl = y_target_lower[2:end]
    
    return vcat(vcat(final_xu, final_xl), vcat(final_yu, final_yl))
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
    p4 = scatter(m_lst, p_lst, t_lst, xlabel="Max Camber", 
                 ylabel="Camber Position", zlabel="Thickness", title="Original Naca Values",label="")
    p5 = scatter(lx,ly,lz, xlabel="Max Camber", 
                 ylabel="Camber Position", zlabel="Thickness", title="Encoded Values", label = "")
    # Save them all together
    savefig(p1, "NACA_encoder/figures/plotted_max_camber_correlation.png")
    savefig(p2, "NACA_encoder/figures/plotted_camber_positon_correlation.png")
    savefig(p3, "NACA_encoder/figures/plotted_thickness_correlation.png")
    savefig(p4, "NACA_encoder/figures/plotted_original_NACA_values.png")
    savefig(p5, "NACA_encoder/figures/plotted_encoded_values.png")
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
        Lux.Dense(central_neurons => central_neurons, relu),
        Lux.Dense(central_neurons => n_digits)
    )
    decoder = Lux.Chain(
        Lux.Dense(n_digits => central_neurons, relu),
        Lux.Dense(central_neurons => central_neurons, relu),
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

current_model = "NACA6V3"
n_coords = 40

function foil()
    nacas = NACA_numbers(5)
    naca_data = get_naca_data(nacas,n_coords)
    model, ps, st, stats_matrix = load_neural_network("NACA_encoder/models/$current_model.jld2")
    naca_data2 = predict_naca_data(naca_data, model, ps, st, stats_matrix, n_coords)
    points = n_coords*2-1
    for i in 1:5
        naca_data_x = naca_data[1:points, i]
        naca_data_y = naca_data[points+1:end, i]
        naca_data2_x = naca_data2[1:points, i]
        naca_data2_y = naca_data2[points+1:end, i]
        if i == 1
            data_lst = zeros(2, points)
            data_lst[1,1:points]=naca_data_x
            data_lst[2,1:points]=naca_data_y
            writedlm("NACA_encoder/airfoil_data/random_NACA.data", transpose(data_lst), "  ")
        end
        plt = plot(naca_data_x, naca_data_y, title="Airfoil Comparison: $(nacas[i])", label="Actual", aspect_ratio=:equal)
        plot!(naca_data2_x, naca_data2_y,label="Decoded")
        savefig(plt, "NACA_encoder/figures/airfoil_sample_$i.png")
    end
end

function new_foils()
    airfoil_lst = ["S2062", "S4022","NACA2412","NACA4424", "random_NACA", "lrn1007", "SD7062", "USA33"]
    airfoils = size(airfoil_lst,1)
    naca_data = zeros((n_coords*4-2),airfoils)
    for i in 1:airfoils
        data = process_general_airfoil("NACA_encoder/airfoil_data/$(airfoil_lst[i]).data", n_coords)
        naca_data[:,i] = data
    end
    model, ps, st, stats_matrix = load_neural_network("NACA_encoder/models/$current_model.jld2")
    naca_data2 = predict_naca_data(naca_data, model, ps, st, stats_matrix, n_coords)
    points = n_coords*2-1
    for i in 1:airfoils
        naca_data_x = naca_data[1:points, i]
        naca_data_y = naca_data[points+1:end, i]
        naca_data2_x = naca_data2[1:points, i]
        naca_data2_y = naca_data2[points+1:end, i]
        plt = plot(naca_data_x, naca_data_y, title="Airfoil Comparison: $(airfoil_lst[i])", label="Actual", aspect_ratio=:equal)
        plot!(naca_data2_x, naca_data2_y,label="Decoded")
        savefig(plt, "NACA_encoder/figures/airfoil_sample_$(airfoil_lst[i]).png")
    end
end

function main(command::String="")
    n_naca = 5000 # How many different NACA foils we will train with
    n_digits = 4 # Digits we would like to encode down to
    nacas = NACA_numbers(n_naca) # Gives n different random NACA airfoils
    naca_data = get_naca_data(nacas,n_coords)
    if command == "train"
        epochs = 600
        model, ps, st, stats_matrix = train_neural_network(n_naca, n_coords, n_digits, naca_data, epochs)
    else
        model, ps, st, stats_matrix = load_neural_network("NACA_encoder/models/$current_model.jld2")
    end
    visualize_code(model, ps, st, nacas, n_naca, n_coords, naca_data, stats_matrix)
end

