function matlab_benchs(trials)
    warning off;
    for alg=["nls", "nlsr", "als", "alsr", "minf", "minfr", "fLMa", "opt"]
        test_alg(alg, trials);
    end
end


function test_alg(alg, trials)
    for x=["swimmer", "border_rank", "mn", "swamp", "bottleneck"]
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
        values = [[5,10], 50:50:50];
        for maxiter=values  
            msg = "Testing " + x + " for " + alg + " with maxiter = " + num2str(maxiter) + "\n";
            fprintf(msg);
            if x=="swamp" || x=="bottleneck"
                if alg=="opt"
                    [best_error, best_time] = tensortoolbox_noise_benchs(T, T_noise, R, tfx_error, trials, alg, maxiter);
                elseif alg=="fLMa"
                    [best_error, best_time] = tensorbox_noise_benchs(T, T_noise, R, tfx_error, trials, alg, maxiter);
                else  
                    [best_error, best_time] = tensorlab_noise_benchs(T, T_noise, R, tfx_error, trials, alg, maxiter);
                end
            else 
                if alg=="opt"
                    [best_error, best_time] = tensortoolbox_benchs(T, R, tfx_error, trials, alg, maxiter);
                elseif alg=="fLMa"
                    [best_error, best_time] = tensorbox_benchs(T, R, tfx_error, trials, alg, maxiter);
                else                
                    [best_error, best_time] = tensorlab_benchs(T, R, tfx_error, trials, alg, maxiter);
                end
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


%TENSORLAB

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
        options.LargeScale = true;
        options.Refinement = false;
    elseif alg=="nlsr"
        options.Algorithm = @cpd_nls;
        options.LargeScale = true;
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


%TENSOR TOOLBOX

function [best_error, best_time] = tensortoolbox_noise_benchs(T, T_noise, R, tfx_error, trials, alg, maxiter)
    best_error = inf;
    best_time = inf;

    options.init = 'randn';
    options.opt = 'lbfgsb';
    options.opt_options.factr = 0;
    options.opt_options.maxIts = maxiter;
    options.opt_options.printEvery = 0;

    T_tensor = tensor(T);
    T_noise_tensor = tensor(T_noise);

    for i=1:trials
        tic;
        W = cp_opt(T_noise_tensor, R, options);
        time = toc;
        T_approx = double(W);
        rel_error = frob(T - T_approx)/frob(T);

        if (rel_error < tfx_error + tfx_error/100) && (rel_error < best_error)
            best_error = rel_error;
            best_time = time;
        end
    end
end

function [best_error, best_time] = tensortoolbox_benchs(T, R, tfx_error, trials, alg, maxiter)
    best_error = inf;
    best_time = inf;

    options.init = 'randn';
    options.opt = 'lbfgsb';
    options.opt_options.factr = 0;
    options.opt_options.maxIts = maxiter;
    options.opt_options.printEvery = 0;

    T_tensor = tensor(T);

    for i=1:trials
        tic;
        W = cp_opt(T_tensor, R, options);
        time = toc;
        T_approx = double(W);
        rel_error = frob(T - T_approx)/frob(T);

        if (rel_error < tfx_error + tfx_error/100) && (rel_error < best_error)
            best_error = rel_error;
            best_time = time;
        end
    end
end


%TENSOR BOX

function [best_error, best_time] = tensorbox_benchs(T, R, tfx_error, trials, alg, maxiter)
    best_error = inf;
    best_time = inf;

    opts.init = 'random';
    opts.tol = 0;
    opts.maxiters = maxiter;
    opts.printitn = 0;

    T_tensor = tensor(T);

    for i=1:trials
        tic;
        W = cp_fLMa(T_tensor, R, opts);
        time = toc;
        T_approx = double(W);
        rel_error = frob(T - T_approx)/frob(T);

        if (rel_error < tfx_error + tfx_error/100) && (rel_error < best_error)
            best_error = rel_error;
            best_time = time;
        end
    end
end

function [best_error, best_time] = tensorbox_noise_benchs(T, T_noise, R, tfx_error, trials, alg, maxiter)
    best_error = inf;
    best_time = inf;

    opts.init = 'random';
    opts.tol = 0;
    opts.maxiters = maxiter;
    opts.printitn = 0;

    T_tensor = tensor(T);
    T_noise_tensor = tensor(T_noise);

    for i=1:trials
        tic;
        W = cp_fLMa(T_noise_tensor, R, opts);
        time = toc;
        T_approx = double(W);
        rel_error = frob(T - T_approx)/frob(T);

        if (rel_error < tfx_error + tfx_error/100) && (rel_error < best_error)
            best_error = rel_error;
            best_time = time;
        end
    end
end
