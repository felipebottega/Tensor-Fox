function matlab_mix_benchs(k, trials, alg)
    R = k;
    filename = ['gaussian_mix_data.mat'];
    load(filename);

    best_error = inf;
    best_time = inf;
    acc = 0;

    options.Display = false;
    options.Initialization = @cpd_rnd; 
    options.TolFun = 1e-12; 
    options.TolX = 1e-12; 

    for i=1:trials
        tic;
        % Create model
        model = struct;
        model.variables.a = randn(size(M3_approx,1), R);
        model.factors.A = 'a';
        model.factorizations.symm.data = M3_approx;
        model.factorizations.symm.cpd = {'A', 'A', 'A'};
        if alg=="nls"
            sol = ccpd_nls(model, options);
        else
            sol = ccpd_minf(model, options);
        end
        acc = acc + toc;
        W1 = sol{1};
        W2 = sol{1};
        W3 = sol{1};
        save(alg + "_" + num2str(i) + "_1.mat", 'W1');
        save(alg + "_" + num2str(i) + "_2.mat", 'W2');
        save(alg + "_" + num2str(i) + "_3.mat", 'W3');
    end

avg_time = acc/trials;
save(alg + "_time.mat", 'avg_time');
disp(['Average time = ', num2str(avg_time), ' sec'])
end


