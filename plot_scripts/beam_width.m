
m1 = csvread('../experiment_data/beam_width/k_0.txt');
m2 = csvread('../experiment_data/beam_width/k_1.txt');
m3 = csvread('../experiment_data/beam_width/k_2.txt');
m4 = csvread('../experiment_data/beam_width/k_4.txt');
m5 = csvread('../experiment_data/beam_width/k_8.txt');
I = csvread('../experiment_data/beam_width/iter_values.txt');

M = [m1, m2, m3, m4, m5];

plot(I, M, 'LineWidth', 1.5);

xlabel('Iteration');
ylabel('Avr Edit Distance');


lgn = legend({'No Beam', 'K = 1', 'k = 2', 'k = 4', 'k = 8'});
lgn.FontSize = 11;
