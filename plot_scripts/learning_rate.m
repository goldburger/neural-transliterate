m1 = csvread('../experiment_data/learning_rate/alpha_0.001.txt');
m2 = csvread('../experiment_data/learning_rate/alpha_0.005.txt');
m3 = csvread('../experiment_data/learning_rate/alpha_0.01.txt');
m4 = csvread('../experiment_data/learning_rate/alpha_0.02.txt');
m5 = csvread('../experiment_data/learning_rate/alpha_0.04.txt');

res = 200;
M = [m1, m2, m3, m4, m5];
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

xticks([0, 5000, 10000, 15000, 20000, 25000, 30000]);
ylim([0, 3]);
lgn = legend({'\alpha 0.001', '\alpha 0.005', '\alpha 0.01', '\alpha 0.02', '\alpha 0.04'})
lgn.FontSize = 11;
