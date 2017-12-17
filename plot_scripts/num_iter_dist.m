x = csvread('../experiment_data/num_iter/iter_values.txt');
y = csvread('../experiment_data/num_iter/distance_values.txt');

x = x(1:50);
y = y(1:50);

plot(x, y, 'LineWidth', 1.5);

xlabel('Iteration');
ylabel('Avr Edit Distance');