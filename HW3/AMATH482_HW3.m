%% Test 1 - Ideal Case: Image Processing
clear all; close all; clc;

% cam1_1
load('cam1_1.mat') 

numFrames = size(vidFrames1_1,4);

y_mask = 180:480;
x_mask = 300:345;
a_pos1_1 = zeros(numFrames,2);
for j = 1:numFrames
    RGB = vidFrames1_1(y_mask,x_mask,:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.95);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos1_1(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos1_1(:,1), a_pos1_1(:,2),'.r','Markersize', 10)
%     drawnow
end

% cam2_1
load('cam2_1.mat') 

numFrames = size(vidFrames2_1,4);

y_mask = 80:380;
x_mask = 250:340;
a_pos2_1 = zeros(numFrames,2);

for j = 1:numFrames
    RGB = vidFrames2_1(y_mask,x_mask,:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.965);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos2_1(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos2_1(:,1), a_pos2_1(:,2),'.r','Markersize', 10)
%     drawnow
end

% cam3_1
load('cam3_1.mat') 

numFrames = size(vidFrames3_1,4);

y_mask = 210:330;
x_mask = 280:480;
a_pos3_1 = zeros(numFrames,2);

for j = 1:numFrames
    RGB = vidFrames3_1(y_mask,x_mask,:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.9);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos3_1(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos3_1(:,1), a_pos3_1(:,2),'.r','Markersize', 10)
%     drawnow
end

%% Test 1 - Ideal Case: PCA
% Separate x and y components from position matrix
x1_1 = a_pos1_1(:, 1); y1_1 = a_pos1_1(:, 2);
x2_1 = a_pos2_1(:, 1); y2_1 = a_pos2_1(:, 2);
x3_1 = a_pos3_1(:, 1); y3_1 = a_pos3_1(:, 2); 

% Make the initial start the same
[~,y1_1_min] = min(y1_1(1:50));
[~,y2_1_min] = min(y2_1(1:50));
[~,x3_1_max] = max(x3_1(1:50));

x1_1 = x1_1(y1_1_min:end); y1_1 = y1_1(y1_1_min:end);
x2_1 = x2_1(y2_1_min:end); y2_1 = y2_1(y2_1_min:end);
x3_1 = x3_1(x3_1_max:end); y3_1 = y3_1(x3_1_max:end);

% Make number of snapshots equal to the minimum
min_snap = min([length(x1_1), length(x2_1), length(x3_1)]);

x1_1 = x1_1(1:min_snap); y1_1 = y1_1(1:min_snap);
x2_1 = x2_1(1:min_snap); y2_1 = y2_1(1:min_snap);
x3_1 = x3_1(1:min_snap); y3_1 = y3_1(1:min_snap);

% Find the average value of the x and y coordinate for each camera to
% prevent the SVD from weighting absolute location
a_x1_1 = mean(x1_1); a_y1_1 = mean(y1_1);
a_x2_1 = mean(x2_1); a_y2_1 = mean(y2_1);
a_x3_1 = mean(x3_1); a_y3_1 = mean(y3_1);

% Subtract each row element from its corresponding average
for n = 1:length(x1_1)
    x1_1(n) = x1_1(n) - a_x1_1; y1_1(n) = y1_1(n) - a_y1_1;
    x2_1(n) = x2_1(n) - a_x2_1; y2_1(n) = y2_1(n) - a_y2_1;
    x3_1(n) = x3_1(n) - a_x3_1; y3_1(n) = y3_1(n) - a_y3_1;
end

% Variables and snapshots collated into a single matrix
A = [x1_1';y1_1';x2_1';y2_1';x3_1';y3_1'];

% Take SVD of collated matrix withan economy decomposition
[U,S,V] = svd(A,'econ');

% Find the energy of the singular values by dividing each singular 
% value by the average of all of the singular values
S_col = diag(S);

for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)*S_col(j,1);
end

S_sum = sum(S_col);
for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)/S_sum;
end
S_diag = diag(S_col);

% Singular value energy plot
figure(1)
subplot(2,2,[2, 4])
plot(1:length(S_col), S_col, 'o', 'LineWidth', 2)
set(gca,'Fontsize',12) 
xlabel('Singular Values');
ylabel('Relative Energy');
title('Relative Energy of Singular Values - Test 1');

% plot of all camera angles showing major variation direction - z
subplot(2,2,1)
plot(1:length(y1_1),y1_1,1:length(y1_1),y2_1,1:length(y1_1),x3_1,'LineWidth',1)
set(gca,'Xlim',[0,length(y1_1)],'Ylim',[-150,150],'Fontsize',12) 
legend("Cam 1","Cam 2","Cam 3")
xlabel('Time (frames)');
ylabel('Displacement (pixels)');
title('Bucket Displacement in Time - Test 1');

% plot of data projections onto pricipal components  
subplot(2,2,3)
A_proj = U'*A;
plot(1:length(A_proj(1,:)), A_proj(1,:),'LineWidth',1)
hold on
plot(1:length(A_proj(2,:)), A_proj(2,:),'LineWidth',1)
hold on
plot(1:length(A_proj(3,:)), A_proj(3,:),'LineWidth',1)
hold on
legend("P.C. 1","P.C. 2","P.C. 3")
set(gca,'Xlim',[0,length(A_proj(1,:))],'Fontsize',12) 
xlabel('Time (frames)');
ylabel('Displacement (pixels)');
title('Data Projections onto Principal Components - Test 1');

%% Test 2 - Noisy Case: Image processing
clear all; close all; clc;

% cam1_2
load('cam1_2.mat') 

numFrames = size(vidFrames1_2,4);

y_mask = 180:480;
x_mask = 280:400;
a_pos1_2 = zeros(numFrames,2);
for j = 1:numFrames
    RGB = vidFrames1_2((y_mask),(x_mask),:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.9999999999);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos1_2(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos1_2(:,1), a_pos1_2(:,2),'.r','Markersize', 10)
%     drawnow
end

% cam2_2
load('cam2_2.mat') 

numFrames = size(vidFrames2_2,4);

y_mask = 40:420;
x_mask = 200:460;
a_pos2_2 = zeros(numFrames,2);
for j = 1:numFrames
    RGB = vidFrames2_2((y_mask),(x_mask),:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.95);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos2_2(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos2_2(:,1), a_pos2_2(:,2),'.r','Markersize', 10)
%     drawnow
end

% cam3_2
load('cam3_2.mat') 

numFrames = size(vidFrames3_2,4);

y_mask = 200:320;
x_mask = 300:500;
a_pos3_2 = zeros(numFrames,2);
for j = 1:numFrames
    RGB = vidFrames3_2((y_mask),(x_mask),:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.95);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos3_2(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos3_2(:,1), a_pos3_2(:,2),'.r','Markersize', 10)
%     drawnow
end

%% Test 2 - Noisy Case: PCA
% Separate x and y components from position matrix
x1_2 = a_pos1_2(:, 1); y1_2 = a_pos1_2(:, 2);
x2_2 = a_pos2_2(:, 1); y2_2 = a_pos2_2(:, 2);
x3_2 = a_pos3_2(:, 1); y3_2 = a_pos3_2(:, 2); 

% Make the initial start the same
[~,y1_2_min] = min(y1_2(1:50));
[~,y2_2_min] = min(y2_2(1:50));
[~,x3_2_max] = max(x3_2(1:50));

x1_2 = x1_2(y1_2_min:end); y1_2 = y1_2(y1_2_min:end);
x2_2 = x2_2(y2_2_min:end); y2_2 = y2_2(y2_2_min:end);
x3_2 = x3_2(x3_2_max:end); y3_2 = y3_2(x3_2_max:end);

% Make number of snapshots equal to the minimum
min_snap = min([length(x1_2), length(x2_2), length(x3_2)]);

x1_2 = x1_2(1:min_snap); y1_2 = y1_2(1:min_snap);
x2_2 = x2_2(1:min_snap); y2_2 = y2_2(1:min_snap);
x3_2 = x3_2(1:min_snap); y3_2 = y3_2(1:min_snap);

% Find the average value of the x and y coordinate for each camera to
% prevent the SVD from weighting absolute location
a_x1_2 = mean(x1_2); a_y1_2 = mean(y1_2);
a_x2_2 = mean(x2_2); a_y2_2 = mean(y2_2);
a_x3_2 = mean(x3_2); a_y3_2 = mean(y3_2);

% Subtract each row element from its corresponding average
for n = 1:length(x1_2)
    x1_2(n) = x1_2(n) - a_x1_2; y1_2(n) = y1_2(n) - a_y1_2;
    x2_2(n) = x2_2(n) - a_x2_2; y2_2(n) = y2_2(n) - a_y2_2;
    x3_2(n) = x3_2(n) - a_x3_2; y3_2(n) = y3_2(n) - a_y3_2;
end

% Variables and snapshots collated into a single matrix
A = [x1_2';y1_2';x2_2';y2_2';x3_2';y3_2'];

% Take SVD of collated matrix withan economy decomposition
[U,S,V] = svd(A,'econ');

% Find the energy of the singular values by dividing each singular 
% value by the average of all of the singular values
S_col = diag(S);

for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)*S_col(j,1);
end

S_sum = sum(S_col);
for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)/S_sum;
end
S_diag = diag(S_col);

% Singular value energy plot
figure(1)
subplot(2,2,[2, 4])
plot(1:length(S_col), S_col, 'o', 'LineWidth', 2)
set(gca,'Fontsize',12) 
xlabel('Singular Values');
ylabel('Relative Energy');
title('Relative Energy of Singular Values - Test 2');

% plot of all camera angles showing major variation direction - z
subplot(2,2,1)
plot(1:length(y1_2),y1_2,1:length(y1_2),y2_2,1:length(y1_2),x3_2,'LineWidth',1)
set(gca,'Xlim',[0,length(y1_2)],'Ylim',[-150,150],'Fontsize',12) 
legend("Cam 1","Cam 2","Cam 3")
xlabel('Time (frames)');
ylabel('Displacement (pixels)');
title('Bucket Displacement in Time - Test 2');

% plot of data projections onto pricipal components  
subplot(2,2,3)
A_proj = U'*A;
plot(1:length(A_proj(1,:)), A_proj(1,:),'LineWidth',1)
hold on
plot(1:length(A_proj(2,:)), A_proj(2,:),'LineWidth',1)
hold on
plot(1:length(A_proj(3,:)), A_proj(3,:),'LineWidth',1)
hold on
plot(1:length(A_proj(4,:)), A_proj(4,:),'LineWidth',1)
hold on
legend("P.C. 1","P.C. 2","P.C. 3","P.C. 4")
set(gca,'Xlim',[0,length(A_proj(1,:))],'Fontsize',12) 
xlabel('Time (frames)');
ylabel('Displacement (pixels)');
title('Data Projections onto Principal Components - Test 2');

%% Test 3 - Horizontal Displacement: Image processing
clear all; close all; clc;

% cam1_3
load('cam1_3.mat') 

numFrames = size(vidFrames1_3,4);

y_mask = 220:480;
x_mask = 280:400;
a_pos1_3 = zeros(numFrames,2);
for j = 1:numFrames
    RGB = vidFrames1_3((y_mask),(x_mask),:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.85);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos1_3(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos1_3(:,1), a_pos1_3(:,2),'.r','Markersize', 10)
%     drawnow
end

% cam2_3
load('cam2_3.mat') 

numFrames = size(vidFrames2_3,4);

y_mask = 180:420;
x_mask = 180:460;
a_pos2_3 = zeros(numFrames,2);
for j = 1:numFrames
    RGB = vidFrames2_3((y_mask),(x_mask),:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.985);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos2_3(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos2_3(:,1), a_pos2_3(:,2),'.r','Markersize', 10)
%     drawnow
end

% cam3_3
load('cam3_3.mat') 

numFrames = size(vidFrames3_3,4);

y_mask = 200:320;
x_mask = 300:500;
a_pos3_3 = zeros(numFrames,2);
for j = 1:numFrames
    RGB = vidFrames3_3((y_mask),(x_mask),:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.95);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos3_3(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos3_3(:,1), a_pos3_3(:,2),'.r','Markersize', 10)
%     drawnow
end

%% Test 3 - Horizontal displacement: PCA
% Separate x and y components from position matrix
x1_3 = a_pos1_3(:, 1); y1_3 = a_pos1_3(:, 2);
x2_3 = a_pos2_3(:, 1); y2_3 = a_pos2_3(:, 2);
x3_3 = a_pos3_3(:, 1); y3_3 = a_pos3_3(:, 2); 

% Make the initial start the same
[~,y1_3_min] = min(y1_3(1:50));
[~,y2_3_min] = min(y2_3(1:50));
[~,x3_3_max] = max(x3_3(1:50));

x1_3 = x1_3(y1_3_min:end); y1_3 = y1_3(y1_3_min:end);
x2_3 = x2_3(y2_3_min:end); y2_3 = y2_3(y2_3_min:end);
x3_3 = x3_3(x3_3_max:end); y3_3 = y3_3(x3_3_max:end);

% Make number of snapshots equal to the minimum
min_snap = min([length(x1_3), length(x2_3), length(x3_3)]);

x1_3 = x1_3(1:min_snap); y1_3 = y1_3(1:min_snap);
x2_3 = x2_3(1:min_snap); y2_3 = y2_3(1:min_snap);
x3_3 = x3_3(1:min_snap); y3_3 = y3_3(1:min_snap);

% Find the average value of the x and y coordinate for each camera to
% prevent the SVD from weighting absolute location
a_x1_3 = mean(x1_3); a_y1_3 = mean(y1_3);
a_x2_3 = mean(x2_3); a_y2_3 = mean(y2_3);
a_x3_3 = mean(x3_3); a_y3_3 = mean(y3_3);

% Subtract each row element from its corresponding average
for n = 1:length(x1_3)
    x1_3(n) = x1_3(n) - a_x1_3; y1_3(n) = y1_3(n) - a_y1_3;
    x2_3(n) = x2_3(n) - a_x2_3; y2_3(n) = y2_3(n) - a_y2_3;
    x3_3(n) = x3_3(n) - a_x3_3; y3_3(n) = y3_3(n) - a_y3_3;
end

% Variables and snapshots collated into a single matrix
A = [x1_3';y1_3';x2_3';y2_3';x3_3';y3_3'];

% Take SVD of collated matrix withan economy decomposition
[U,S,V] = svd(A,'econ');

% Find the energy of the singular values by dividing each singular 
% value by the average of all of the singular values
S_col = diag(S);

for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)*S_col(j,1);
end

S_sum = sum(S_col);
for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)/S_sum;
end
S_diag = diag(S_col);

% Singular value energy plot
figure(1)
subplot(2,2,[2, 4])
plot(1:length(S_col), S_col, 'o', 'LineWidth', 2)
set(gca,'Fontsize',12) 
xlabel('Singular Values');
ylabel('Relative Energy');
title('Relative Energy of Singular Values - Test 3');

% plot of all camera angles showing major variation direction - z
subplot(2,2,1)
plot(1:length(y1_3),y1_3,1:length(y1_3),y2_3,1:length(y1_3),x3_3,'LineWidth',1)
set(gca,'Xlim',[0,length(y1_3)],'Ylim',[-100,100],'Fontsize',12) 
legend("Cam 1","Cam 2","Cam 3")
xlabel('Time (frames)');
ylabel('Displacement (pixels)');
title('Bucket Displacement in Time - Test 3');

% plot of data projections onto pricipal components  
subplot(2,2,3)
A_proj = U'*A;
plot(1:length(A_proj(1,:)), A_proj(1,:),'LineWidth',1)
hold on
plot(1:length(A_proj(2,:)), A_proj(2,:),'LineWidth',1)
hold on
plot(1:length(A_proj(3,:)), A_proj(3,:),'LineWidth',1)
hold on
plot(1:length(A_proj(4,:)), A_proj(4,:),'LineWidth',1)
hold on
legend("P.C. 1","P.C. 2","P.C. 3","P.C. 4")
set(gca,'Xlim',[0,length(A_proj(1,:))],'Fontsize',12) 
xlabel('Time (frames)');
ylabel('Displacement (pixels)');
title('Data Projections onto Principal Components - Test 3');

%% Test 4 - Horizontal Displacement and Rotation: Image processing
clear all; close all; clc;

% cam1_4
load('cam1_4.mat') 

numFrames = size(vidFrames1_4,4);

y_mask = 230:440;
x_mask = 300:440;
a_pos1_4 = zeros(numFrames,2);
for j = 1:numFrames
    RGB = vidFrames1_4((y_mask),(x_mask),:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.675);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos1_4(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos1_4(:,1), a_pos1_4(:,2),'.r','Markersize', 10)
%     drawnow
end

% cam2_4
load('cam2_4.mat') 

numFrames = size(vidFrames2_4,4);

y_mask = 100:360;
x_mask = 200:450;
a_pos2_4 = zeros(numFrames,2);
for j = 1:numFrames
    RGB = vidFrames2_4((y_mask),(x_mask),:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.985);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos2_4(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos2_4(:,1), a_pos2_4(:,2),'.r','Markersize', 10)
%     drawnow
end

% cam3_4
load('cam3_4.mat') 

numFrames = size(vidFrames3_4,4);

y_mask = 120:300;
x_mask = 320:520;
a_pos3_4 = zeros(numFrames,2);
for j = 1:numFrames
    RGB = vidFrames3_4((y_mask),(x_mask),:,j);
    gray = rgb2gray(RGB);
    gray_b = imbinarize(gray, 0.9);
    w_pixel = find(gray_b);
    sum_pos = [0,0];
    for jj = (1:length(w_pixel))
        [y,x] = ind2sub(size(gray_b), w_pixel(jj));
    end
    a_pos3_4(j,:) = [mean(x),mean(y)];
%     imshow(gray_b)
%     hold on
%     plot(a_pos3_4(:,1), a_pos3_4(:,2),'.r','Markersize', 10)
%     drawnow
end

%% Test 4 - Horizontal displacement and Rotation: PCA
% Separate x and y components from position matrix
x1_4 = a_pos1_4(:, 1); y1_4 = a_pos1_4(:, 2);
x2_4 = a_pos2_4(:, 1); y2_4 = a_pos2_4(:, 2);
x3_4 = a_pos3_4(:, 1); y3_4 = a_pos3_4(:, 2); 

% Make the initial start the same
[~,y1_4_min] = min(y1_4(1:50));
[~,y2_4_min] = min(y2_4(1:50));
[~,x3_4_max] = max(x3_4(1:50));

x1_4 = x1_4(y1_4_min:end); y1_4 = y1_4(y1_4_min:end);
x2_4 = x2_4(y2_4_min:end); y2_4 = y2_4(y2_4_min:end);
x3_4 = x3_4(x3_4_max:end); y3_4 = y3_4(x3_4_max:end);

% Make number of snapshots equal to the minimum
min_snap = min([length(x1_4), length(x2_4), length(x3_4)]);

x1_4 = x1_4(1:min_snap); y1_4 = y1_4(1:min_snap);
x2_4 = x2_4(1:min_snap); y2_4 = y2_4(1:min_snap);
x3_4 = x3_4(1:min_snap); y3_4 = y3_4(1:min_snap);

% Find the average value of the x and y coordinate for each camera to
% prevent the SVD from weighting absolute location
a_x1_4 = mean(x1_4); a_y1_4 = mean(y1_4);
a_x2_4 = mean(x2_4); a_y2_4 = mean(y2_4);
a_x3_4 = mean(x3_4); a_y3_4 = mean(y3_4);

% Subtract each row element from its corresponding average
for n = 1:length(x1_4)
    x1_4(n) = x1_4(n) - a_x1_4; y1_4(n) = y1_4(n) - a_y1_4;
    x2_4(n) = x2_4(n) - a_x2_4; y2_4(n) = y2_4(n) - a_y2_4;
    x3_4(n) = x3_4(n) - a_x3_4; y3_4(n) = y3_4(n) - a_y3_4;
end

% Variables and snapshots collated into a single matrix
A = [x1_4';y1_4';x2_4';y2_4';x3_4';y3_4'];

% Take SVD of collated matrix withan economy decomposition
[U,S,V] = svd(A,'econ');

% Find the energy of the singular values by dividing each singular 
% value by the average of all of the singular values
S_col = diag(S);

for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)*S_col(j,1);
end

S_sum = sum(S_col);
for j = 1:length(S_col)
    S_col(j,1) = S_col(j,1)/S_sum;
end
S_diag = diag(S_col);

% Singular value energy plot
figure(1)
subplot(2,2,[2, 4])
plot(1:length(S_col), S_col, 'o', 'LineWidth', 2)
set(gca,'Fontsize',12) 
xlabel('Singular Values');
ylabel('Relative Energy');
title('Relative Energy of Singular Values - Test 4');

% plot of all camera angles showing major variation direction - z
subplot(2,2,1)
plot(1:length(y1_4),y1_4,1:length(y1_4),y2_4,1:length(y1_4),x3_4,'LineWidth',1)
set(gca,'Xlim',[0,length(y1_4)],'Ylim',[-150,150],'Fontsize',12) 
legend("Cam 1","Cam 2","Cam 3")
xlabel('Time (frames)');
ylabel('Displacement (pixels)');
title('Bucket Displacement in Time - Test 4');

% plot of data projections onto pricipal components  
subplot(2,2,3)
A_proj = U'*A;
plot(1:length(A_proj(1,:)), A_proj(1,:),'LineWidth',1)
hold on
plot(1:length(A_proj(2,:)), A_proj(2,:),'LineWidth',1)
hold on
plot(1:length(A_proj(3,:)), A_proj(3,:),'LineWidth',1)
hold on
plot(1:length(A_proj(4,:)), A_proj(4,:),'LineWidth',1)
hold on
legend("P.C. 1","P.C. 2","P.C. 3","P.C. 4")
set(gca,'Xlim',[0,length(A_proj(1,:))],'Fontsize',12) 
xlabel('Time (frames)');
ylabel('Displacement (pixels)');
title('Data Projections onto Principal Components - Test 4');