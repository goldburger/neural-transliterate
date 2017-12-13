m1 = csvread('../experiment_data/teacher_forcing/teacher_forcing_0.0.txt');
m2 = csvread('../experiment_data/teacher_forcing/teacher_forcing_0.1.txt');
m3 = csvread('../experiment_data/teacher_forcing/teacher_forcing_0.5.txt');
m4 = csvread('../experiment_data/teacher_forcing/teacher_forcing_0.9.txt');
m5 = csvread('../experiment_data/teacher_forcing/teacher_forcing_1.0.txt');

res = 200;
M = [m2, m3, m4];
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

%lgn = legend({'TF 0.0', 'TF 0.1', 'TF 0.5', 'TF 0.9', 'TF 1.0'});
lgn = legend({'TF = 0.1', 'TF = 0.5', 'TF = 0.9'});
lgn.FontSize = 11;
