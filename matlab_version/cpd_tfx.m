function [factors, T_approx, output] = cp_tfx(T, R, options, load_results)
    % Matlab wrapper for the CP decomposition function of Tensor Fox.
    % By default this function assumes that the user will pass the tensor as an array without saving it to the disk 
    % manually. If the tensor originally comes from a file, you should pass T as a string with the filename where the 
    % tensor is stored. Otherwise the tensor will be saved and loaded again, increasing the computational time. 
    % If there is no necessity of working with the results in Matlab, set load_results to 'false', because by
    % default all results are saved to the disk anyway.
    %
    % Inputs
    % ------
    % T: string or array or tensor format
    %     If T is a string, then it must be the filename where the tensor is stored. If T has tensor format, then it is
    %     converted to an array. 
    % R: int
    %     Desired rank.
    % options: structure
    %    maxiter: int
    %        Number of maximum iterations allowed for the dGN function. Default is 200.
    %    tol, tol_step, tol_improv, tol_grad: float
    %        Tolerance criterion to stop the iteration process of the dGN function. Default is 1e-6 for all. Let T^(k) be
    %        the approximation at the k-th iteration, with corresponding CPD w^(k) in vectorized form. The program stops 
    %        if
    %            1) |T - T^(k)| / |T| < tol
    %            2) | w^(k-1) - w^(k) | < tol_step
    %            3) | |T - T^(k-1)| / |T| - |T - T^(k)| / |T| | < tol_improv
    %            4) | grad F(w^(k)) | < tol_grad, where F(w^(k)) = 1/2 |T - T^(k)|^2
    %    tol_mlsvd: float
    %        Tolerance criterion for the truncation. The idea is to obtain a truncation (U_1,...,U_L)*S such that
    %        |T - (U_1,...,U_L)*S| / |T| < tol_mlsvd. Default is 1e-16. There are also two special cases:
    %            1) tol_mlsvd = 0: compress the tensor (that is, compute its MLSVD) but do not truncate the central
    %            tensor of the MLSVD.
    %            2) tol_mlsvd = -1: use the original tensor, so the computation of the MLSVD is not performed.
    %    trunc_dims: int or list of ints
    %        Consider a three order tensor T. If trunc_dims is not 0, then it should be a list with three integers
    %        [R1,R2,R3] such that 1 <= R1 <= m, 1 <= R2 <= n, 1 <= R3 <= p. The compressed tensor will have dimensions
    %        (R1,R2,R3). Default is 0, which means 'automatic' truncation.
    %    initialization: string or list
    %        This options is used to choose the initial point to start the iterations. For more information, check the 
    %        function starting_point.
    %    refine: bool
    %        If True, after the dGN iterations the program uses the solution to repeat the dGN over the original space
    %        using the solution as starting point. Default is False.
    %    symm: bool
    %        The user should set symm to True if the objective tensor is symmetric, otherwise symm is False. Default is
    %        False.
    %    low, upp, factor: floats
    %        These values sets constraints to the entries of the tensor. Default for all of them is 0, which means no 
    %        restriction. The parameter factor is auxiliar and influences how tight are the projections into the 
    %        interval [low, upp]. These parameters are experimental.
    %    trials: int
    %        This parameter is only used for tensor with order higher than 3. The computation of the tensor train CPD 
    %        requires the computation of several CPD of third order tensors. If only one of these CPD's is of low 
    %        quality (divergence or local minima) then all effort is in vain. One work around is to compute several 
    %        CPD'd and keep the best, for third order tensor. The parameter trials defines the maximum number of
    %        times we repeat the computation of each third order CPD. These trials stops when the relative error is
    %        less than 1e-4 or when the maximum number of trials is reached. Default is trials=1.
    %    display: -2, -1, 0, 1, 2, 3 or 4
    %        This options is used to control how information about the computations are displayed on the screen. The 
    %        possible values are -1, 0, 1 (default), 2, 3, 4. Notice that display=3 makes the overall running time large
    %        since it will force the program to show intermediate errors which are computationally costly. -1 is a
    %        special option for displaying minimal relevant information for tensors with order higher then 3. We
    %        summarize the display options below.
    %            -2: display same as options -1 plus the Tensor Train error
    %            -1: display only the errors of each CPD computation and the final relevant information
    %            0: no information is printed
    %            1: partial information is printed
    %            2: full information is printed
    %            3: full information + errors of truncation and starting point are printed
    %            4: almost equal to display = 3 but now there are more digits displayed on the screen (display = 3 is a
    %            "cleaner" version of display = 4, with less information).
    %    epochs: int
    %        Number of Tensor Train CPD cycles. Use only for tensor with order higher than 3. Default is epochs=1.
    %
    % It is not necessary to create 'options' with all parameters described above. Any missing parameter is assigned to
    % its default value automatically. For more information about the options, check the Tensor Fox tutorial at    
    % https://github.com/felipebottega/Tensor-Fox/tree/master/tutorial
    %
    % load_results: bool
    %     If true (default) then all results are saved and loaded to the current Matlab session.
    %
    % Outputs (if load_results is true)
    % ---------------------------------
    % factors: ktensor
    %     Each array factors{i} correspond to the i-th factor matrix of the approximated CPD of T.
    % T_approx: array
    %     Approximated tensor in coordinate format.
    % output: structure
    %     Structure with all relevant information obtained during the computation of the CPD.
    %
    % REFERENCES
    % ----------
    % https://github.com/felipebottega/Tensor-Fox

    % Initial verifications.     
    if ~exist("options", "var")
        % Third parameter does not exist, so default it to a simple structure with known default value.
        options.maxiter = 200;
    end

    if ~exist("load_results", "var")
        % Fourth parameter does not exist, so default it to 1 (true).
        load_results = 1;
    end

    if isa(T, 'tensor')
        % In the case the original tensor is in array format, don't convert it to tensor format. Pass the array instead.
        T = double(T);
    end

    % Get path to tensorfox folder and file.
    % I will be something like path_cp_tfx = "/home/usr/Documents/tensor_toolbox-v3.1/cp_tfx.py".
    % We also have path_tfx = "/home/usr/Documents/tensor_toolbox-v3.1/tensorfox".
    path_cp_tfx = mfilename('fullpath');
    % Remove the "cp_tfx" from the string.
    path_tfx = path_cp_tfx(1:length(path_cp_tfx)-7);
    % Add "tensorfox" to the end of the string;
    path_tfx = path_tfx + "tensorfox";
    % Get path to Python file "cp_tfx.py". 
    path_cp_tfx = path_cp_tfx + ".py";

    % Get path of the current workspace.
    path_ws = pwd;

    % If T is an array, the program saves the tensor to the disk. Otherwise, it is assumed that the
    % variable T is the path where the tensor is stored.
    if isfloat(T)
        tensor_path = fullfile(path_ws, 'T.mat');
        save(tensor_path, 'T', '-v7.3', '-nocompression')
    else
        tensor_path = T;
    end

    % Save options to the disk. 
    options_path = fullfile(path_ws, 'options.mat');
    save(options_path, "options");
    
    % Make directory to receive outputs.
    warning off;
    mkdir outputs;

    % Call Tensor Fox from terminal.
    % The following two lines should be used for enviroment usage, changing 'base' to the enviroment name.
    % command_line = "!conda activate base";
    % eval(command_line)
    if load_results
        command_line = "!python " + path_cp_tfx + " " + tensor_path + " " + num2str(R) + " " + options_path + " 1 " + path_tfx;
    else
        command_line = "!python " + path_cp_tfx + " " + tensor_path + " " + num2str(R) + " " + options_path + " 0 " + path_tfx;
    end
    eval(command_line)
    fprintf(repmat('\b',1,4));
    %fprintf('\n');

    % Load the saved results of Tensor Fox. 
    if load_results
        % Factor matrices.
        path_output = fullfile(path_ws, 'outputs', 'factors.mat');
        factors_struct = load(path_output);
        fn = fieldnames(factors_struct);
        % Because HDF5 stores data in row-major order and the MATLAB array is organized in column-major order, you should 
        % reverse the ordering of the dimension, so the loop is backwards.
        i = 1;
        for l=numel(fn):-1:1
            factors{i} = factors_struct.(fn{l});
            i = i+1;
        end
        factors = ktensor(factors);
        % Arrange the final tensor so that the columns are normalized.
        factors = arrange(factors);
        % Fix the signs
        factors = fixsigns(factors);
        % Coordinate approximate tensor. 
        T_approx = double(factors);
        % Output structure.
        path_output = fullfile(path_ws, 'outputs', 'output.mat');
        output = load(path_output);
    end

