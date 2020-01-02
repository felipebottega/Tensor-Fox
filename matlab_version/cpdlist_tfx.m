function [errors, timings] = cplist_tfx(tensor_paths, ranks_path, options_paths, save_results)
    % Matlab wrapper to the CPD function of Tensor Fox called several times in a sequence.
    % By default no output is saved to the disk, but if this is the desired behavior, set the
    % parameter save_results to 'true'. Then, for each tensor in the list, an output in the
    % format [factors, T_approx, output] is saved in a folder called 'outputs' located at the
    % working space of the current Matlab session.
    % We remark that the tensors must be as multidimensional arrays in .mat format.
    %
    % Inputs
    % ------
    % tensor_paths: string 
    %     Path to the text file containing the paths of all tensors. 
    % ranks_path: string
    %     Path to the text file containing the ranks.
    % options_paths: string
    %     Path to the text file containing the paths of all options structures.
    % save_results: bool
    %     If true then all results are saved in the folder 'outputs'. Default is false.
    %
    % Outputs 
    % -------
    % errors: array
    %     The entry errors(i) correspond to the relative error of the CPD of the ith tensor.
    % timings: array
    %     The entry errors(i) correspond to the time elapsed (in seconds) to compute the CPD of the ith tensor.
    %
    % REFERENCES
    % ----------
    % https://github.com/felipebottega/Tensor-Fox

    if ~exist("save_results", "var")
        % Fourth parameter does not exist, so default it to 0 (false).
        save_results = 0;
    end

    % Get path to tensorfox folder and file. 
    % It will be something like "/home/usr/Documents/tensor_toolbox-v3.1/cplist_tfx.py".
    % We also have path_tfx = "/home/usr/Documents/tensor_toolbox-v3.1/tensorfox".
    path_cplist_tfx = mfilename('fullpath');
    % Remove the "cplist_tfx" from the string.
    path_tfx = path_cplist_tfx(1:length(path_cplist_tfx)-10);
    % Add "tensorfox" to the end of the string;
    path_tfx = path_tfx + "tensorfox";
    % Get path to Python file "cplist_tfx.py". 
    path_cplist_tfx = path_cplist_tfx + ".py";

    % Get path of the current workspace.
    path_ws = pwd;
    
    % Make directory to receive outputs.
    warning off;
    mkdir outputs;
    
    % Call Tensor Fox from terminal.
    % The following two lines should be used for enviroment usage.
    % command_line = "!conda activate base";
    % eval(command_line)
    if save_results
        command_line = "!python " + path_cplist_tfx + " " + tensor_paths + " " + ranks_path + " " + options_paths + " 1 " + path_tfx;
    else
        command_line = "!python " + path_cplist_tfx + " " + tensor_paths + " " + ranks_path + " " + options_paths + " 0 " + path_tfx;
    end
    eval(command_line)
    fprintf(repmat('\b',1,4));
    %fprintf('\n');
    
    % Load arrays with relative errors and timings.
    errors_struct = load("errors.mat");
    errors = errors_struct.errors;
    timings_struct = load("timings.mat");
    timings = timings_struct.timings;
