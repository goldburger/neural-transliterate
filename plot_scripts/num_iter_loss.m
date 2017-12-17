x = csvread('../experiment_data/num_iter/iter_values.txt');
y = csvread('../experiment_data/num_iter/loss_curve.txt');

res = 200;
M = y;
[k, n] = size(M);

P = zeros(k/res, n);
I = 1:res:k;

for i = 1:(k/res)
   j = (i-1) * res +1;
   P(i, :) = mean(M(j:(j+res)-1, :));
end

plot(I, P, 'LineWidth', 1.5);

xlabel('Iteration');
ylabel('Avr Loss');