m1 = csvread('../experiment_data/attention/atn_loss_curve.txt');
m2 = csvread('../experiment_data/attention/no_atn_loss_curve.txt');

res = 100;
M = [m1, m2];
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

lgn = legend({'With Attention', 'No Attention'})
lgn.FontSize = 11;
