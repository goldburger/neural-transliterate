
m1 = csvread('../experiment_data/batch_size/bs_1.txt');
m2 = csvread('../experiment_data/batch_size/bs_10.txt');
m3 = csvread('../experiment_data/batch_size/bs_20.txt');
m4 = csvread('../experiment_data/batch_size/bs_40.txt');
m5 = csvread('../experiment_data/batch_size/bs_80.txt');
m6 = csvread('../experiment_data/batch_size/bs_160.txt');

k = 32000;

%% bs_1
res = 160;
[~, n] = size(m1);
P = zeros(k/res, n);
I = 1:res:k;

for i = 1:(k/res)
   j = (i-1) * res +1;
   P(i, :) = mean(m1(j:(j+res)-1, :));
end

plot(I, P, 'LineWidth', 1.5); hold on;

%% bs_10
P = zeros(200, 1);
I = 1:160:k;

for i = 1:200
   j = ((i-1) * 15);
   P(i, :) = mean(m2((j+1):(j+16), :)) / 10.0;
end

plot(I, P, 'LineWidth', 1.5);

%% bs_20
P = zeros(200, 1);
I = 1:160:k;

for i = 1:200
   j = ((i-1) * 7);
   P(i, :) = mean(m3((j+1):(j+8), :)) / 20.0;
end

plot(I, P, 'LineWidth', 1.5);

%% bs_40
P = zeros(200, 1);
I = 1:160:k;

for i = 1:200
   j = ((i-1) * 3);
   P(i, :) = mean(m4((j+1):(j+4), :)) / 40.0;
end

plot(I, P, 'LineWidth', 1.5);

%% bs_80
P = zeros(200, 1);
I = 1:160:k;

for i = 1:200
   j = ((i-1) * 1);
   P(i, :) = mean(m5((j+1):(j+2), :)) / 80.0;
end

plot(I, P, 'LineWidth', 1.5);

%% bs_160
P = zeros(200, 1);
I = 1:160:k;

for i = 1:200
   P(i, :) = m6(i, :) / 160.0;
end

plot(I, P, 'LineWidth', 1.5);

xlabel('Iteration');
ylabel('Loss');


lgn = legend({'m = 1', 'm = 10', 'm = 20', 'm = 40', 'm = 80', 'm = 160'});
lgn.FontSize = 11;
