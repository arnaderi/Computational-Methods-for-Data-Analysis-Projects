clear all; close all; clc;
load Testdata

% spatial domain
L = 15; 
% Fourier modes
n = 64; 
%number of time points in Undata
timepoints = size(Undata,1); 
x2 = linspace(-L,L,n+1); x = x2(1:n); y = x; z = x; 
%freq components for fft scaled to 2 pi
k = (2*pi/(2*L)) * [0:(n/2-1) -n/2:-1]; 
%shifted freq components to be mathematically correct
ks = fftshift(k); 

[X, Y, Z] = meshgrid(x, y, z);
[Kx, Ky, Kz] = meshgrid(ks, ks, ks);

% Go to the frequency domain with fftn, average across time points
avefftUn=zeros(64,64,64);

for j=1:timepoints
    Un(:, :, :) = reshape(Undata(j,:),n,n,n);
    avefftUn(:, :, :) = avefftUn(:, :, :) + fftn(Un);
end

% Average of the frequency domain of Un, now shifted to be mathematically
% correct
avefftUn=fftshift(avefftUn)./timepoints;

% Go back to the spatial domain of Un, to locate frequency space index
aveUn=abs(ifftn(avefftUn));

[maxfreq, maxfreqindex] = max(abs(avefftUn(:)));
[maxKxindex, maxKyindex, maxKzindex] = ind2sub(size(avefftUn), maxfreqindex);

%Identifying the center frequency for each principal axis
Kx0 = Kx(maxKxindex, maxKyindex, maxKzindex);
Ky0 = Ky(maxKxindex, maxKyindex, maxKzindex);
Kz0 = Kz(maxKxindex, maxKyindex, maxKzindex);

%Filter data using a Gaussian filter
tau = 0.2;
gausfilter = exp(-tau*(((Kx-Kx0).^2)+((Ky-Ky0).^2)+((Kz-Kz0).^2)));

%Creating matrices to store the location
x_pos = zeros(1,timepoints);
y_pos = zeros(1,timepoints);
z_pos = zeros(1,timepoints);

%Applying the filter to each timepoint in the frequency domain, must 
%reshape the data again and shift it at each timepoint

for j=1:timepoints
    Un(:, :, :) = reshape(Undata(j,:), n, n, n);
    fftUnfilter = gausfilter.*fftshift(fftn(Un)); 
  	%filtered frequency in spatial domain
    Unfilter = abs(ifftn(fftUnfilter)); 
    
    [maximum, index] = max(Unfilter(:));
    [filterKxindex, filterKyindex, filterKzindex] = ind2sub(size(fftUnfilter), index);
    
    x_pos(1,j) = X(filterKxindex, filterKyindex, filterKzindex);
    y_pos(1,j) = Y(filterKxindex, filterKyindex, filterKzindex);
    z_pos(1,j) = Z(filterKxindex, filterKyindex, filterKzindex);
    
    isosurface(X,Y,Z, Unfilter,0.4), grid on
    hold on
    title('Isosurface Plot of Marble Position Over Time', 'Fontsize', 24)
    xlabel('X axis', 'Fontsize', 24)
    ylabel('Y axis', 'Fontsize', 24)
    zlabel('Z axis', 'Fontsize', 24)
end

figure()
plot3(x_pos, y_pos, z_pos, 'ko')
hold on
plot3(x_pos, y_pos, z_pos)
title('Trajectory of Marble Position', 'Fontsize', 24)
xlabel('X axis', 'Fontsize', 24)
ylabel('Y axis', 'Fontsize', 24)
zlabel('Z axis', 'Fontsize', 24)
axis([-L L -L L -L L]), grid on, drawnow

figure()
isosurface(X, Y, Z, Unfilter, 0.4)
title('Marble Position at 20th Timepoint', 'Fontsize', 24)
xlabel('X axis', 'Fontsize', 24)
ylabel('Y axis', 'Fontsize', 24)
zlabel('Z axis', 'Fontsize', 24)
axis([-L L -L L -L L]), grid on, drawnow

breakup_pos = [x_pos(1, j), y_pos(1, j), z_pos(1, j)]

