function matlab_benchs(trials)
    warning off;
    for alg=["nls", "nlsr", "als", "alsr", "minf", "minfr"]
        test_alg(alg, trials);
    end
end


function test_alg(alg, trials)
    for x=["swimmer", "hw", "border_rank", "mn", "swamp", "bottleneck"]
        % Load tensor and rank.
        test = x + ".mat";
        struct = load(test);
        A = struct.T;
        T = A;
        r = double(struct.R);
        R = r;
        tfx_err = struct.tfx_error;
        tfx_error = tfx_err;
        if x=="swamp" || x=="bottleneck"
            test = x + "_noise.mat";
            struct = load(test);
            A_noise = struct.T_noise;
            T_noise = A_noise;
        end

        % Start benchmarks.
        values = [[5,10], [50:50:1000]];
        for maxiter=values  
            msg = "Testing " + x + " for " + alg + " with maxiter = " + num2str(maxiter) + "\n";
            fprintf(msg);
            if x=="swamp" || x=="bottleneck"
                [best_error, best_time] = tensorlab_noise_benchs(T, T_noise, R, tfx_error, trials, alg, maxiter);
            else 
                [best_error, best_time] = tensorlab_benchs(T, R, tfx_error, trials, alg, maxiter);
            end

            % When best_error < inf, the program achieved an acceptable solution.
            % The file saved is an array of the form [error, time], with the best results of alg over the tensor given.
            if best_error < inf
                results = [best_error, best_time];
                filename = x + '_' + alg;
                save(filename, 'results');
                disp(['    best error = ', num2str(best_error), '    best time = ', num2str(best_time)])
                break;
            end
        end 
    end
end


function [best_error, best_time] = tensorlab_noise_benchs(T, T_noise, R, tfx_error, trials, alg, maxiter)
    best_error = inf;
    best_time = inf;

    options.Display = false;
    options.Initialization = @cpd_rnd; 
    options.AlgorithmOptions.Maxiter = maxiter;
    options.TolFun = 0;
    options.TolX = 0;

    if alg=="nls"
        options.Algorithm = @cpd_nls;
        options.Refinement = false;
    elseif alg=="nlsr"
        options.Algorithm = @cpd_nls;
    elseif alg=="als"
        options.Algorithm = @cpd_als;
        options.Refinement = false;
    elseif alg=="alsr"
        options.Algorithm = @cpd_als;
    elseif alg=="minf"
        options.Algorithm = @cpd_minf;
        options.Refinement = false;
    elseif alg=="minfr"
        options.Algorithm = @cpd_minf;
    end

    for i=1:trials
        tic;
        W = cpd(T_noise, R, options);
        time = toc;
        T_approx = cpdgen(W);
        rel_error = frob(T - T_approx)/frob(T);

        if (rel_error < tfx_error + tfx_error/100) && (rel_error < best_error)
            best_error = rel_error;
            best_time = time;
        end
    end
end


function [best_error, best_time] = tensorlab_benchs(T, R, tfx_error, trials, alg, maxiter)
    best_error = inf;
    best_time = inf;

    options.Display = false;
    options.Initialization = @cpd_rnd; 
    options.AlgorithmOptions.Maxiter = maxiter;

    if alg=="nls"
        options.Algorithm = @cpd_nls;
        options.Refinement = false;
    elseif alg=="nlsr"
        options.Algorithm = @cpd_nls;
    elseif alg=="als"
        options.Algorithm = @cpd_als;
        options.Refinement = false;
    elseif alg=="alsr"
        options.Algorithm = @cpd_als;
    elseif alg=="minf"
        options.Algorithm = @cpd_minf;
        options.Refinement = false;
    elseif alg=="minfr"
        options.Algorithm = @cpd_minf;
    end

    for i=1:trials
        tic;
        W = cpd(T, R, options);
        time = toc;
        T_approx = cpdgen(W);
        rel_error = frob(T - T_approx)/frob(T);

        if (rel_error < tfx_error + tfx_error/100) && (rel_error < best_error)
            best_error = rel_error;
            best_time = time;
        end
    end
end
