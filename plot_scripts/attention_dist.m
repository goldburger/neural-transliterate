m1 = csvread('../experiment_data/attention/attn_dist_curve.txt');
m2 = csvread('../experiment_data/attention/no_attn_dist_curve.txt');
it = csvread('../experiment_data/attention/iter_values.txt');


plot(it, [m1, m2], 'LineWidth', 1.5);

xlabel('Iteration');
ylabel('Avr Edit Distance');

lgn = legend({'With Attention', 'No Attention'})
lgn.FontSize = 11;