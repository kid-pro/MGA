function [rmse_value] = data_RMSE(matrix1, matrix2)
    % Ensure that the input matrices are of the same size
    assert(all(size(matrix1) == size(matrix2)), 'The input matrices must be of the same size');

    % Calculate the root mean square error
    rmse_value = sqrt(  mean((matrix1(:) - matrix2(:)).^2));
end

