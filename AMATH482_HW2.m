%% PART 1
clear all; close all; clc; 
load handel

% Signal of interest, v(n)
v = y';

% Number of sampled points
n = length(v);

% Length of the signal as determined by number of sampled points divided
% by the rate at which they were sampled
L = n/Fs;

% Time component of the signal
v_time_span2 = linspace(0, L, n+1);
v_time_span = v_time_span2(1:n);

%% Frequency Analysis
% Frequency components
k = (2*pi/L)*[0:(length(v)-1)/2 -(length(v)-1)/2:-1];

% Shifted freq components to be mathematically correct; scaled for Hz
% instead of angular frequencies
k_shift_v = fftshift(k)/(2*pi); 

% FFT of signal of interest
v_fft = fft(v);

% Plot the signal of interest's amplitude as a function of time
figure(1)
subplot(2,2,1) 
plot(v_time_span,v,'k','Linewidth',1); axis([0 L -1 1])
set(gca,'Fontsize',12), xlabel('Time (sec)'), ylabel('v(n)'), title("SOI 1 in Time Domain");

% Plot the signal of interest's FFT
subplot(2,2,3) 
plot(k_shift_v,abs(fftshift(v_fft))/max(abs(v_fft)),'r','Linewidth',1); axis([-4100 4100 0 1])
set(gca,'Fontsize',12)
xlabel('Frequency (Hz)'), ylabel('FFT(v)'), title("SOI 1 in Frequency Domain")

% Plot the signal of interest's spectrogram
subplot (2,2,[2,4])
% Time component of the signal of interest
tslide = 0:0.1:L;
% Frequency component of the signal of interest remapped to an array that
% matches the time component's length
v_fft_spec = repmat(fftshift(abs(v_fft)),length(tslide),1);
pcolor(tslide,k_shift_v,v_fft_spec.');
shading interp
set(gca,'Fontsize',12), xlabel('Time (sec)'), ylabel('Frequency (Hz)'), title('Spectrogram from SOI 1')
colormap(hot), colorbar

%% Gabor window added to time domain plot
% Gabor transform parameters:
% Time steps at which the filter is applied
tau = 4;
% Window width
a = 1;
% Filter definition
g = exp(-a*(v_time_span-tau).^2);
% Time components of the filter
tslide = 0:tau:L;
% Allocate space for spectrogram
vg_fft_spec = zeros(length(tslide), length(v));

figure(2)
%Calculate the spectrogram matrix
for j=1:length(tslide)
    g=exp(-a*(v_time_span-tslide(j)).^2); 
    vg=g.*v; 
    subplot(2,2,1)
    plot(v_time_span,v,'k','Linewidth',1);
    set(gca,'Fontsize',12), xlabel('Time (sec)'), ylabel('v(n)'), title("SOI 1 and Gabor Transform in Time Domain");
    hold on
    subplot(2,2,3)
    plot(v_time_span,vg,'m','Linewidth',1), axis([0 L -1 1]);
    set(gca,'Fontsize',12), xlabel('Time (sec)'), ylabel('v(n)'), title("Gabor Filtered SOI 1 in Time Domain")
    hold on
    vg_fft=fft(vg);  
    vg_fft_spec(j,:) = fftshift(abs(vg_fft));
end

% Plotting the Gabor Transforms on top of SOI 1 in Time Domain
for j=1:length(tslide)
    g=exp(-a*(v_time_span-tslide(j)).^2); 
    vg=g.*v; 
    subplot(2,2,1)
    plot(v_time_span, g,'m','Linewidth',1); axis([0 L -1 1]);
end

% Plotting the spectrogram matrix
subplot(2,2,[2,4])
pcolor(tslide,k_shift_v,vg_fft_spec.'), 
shading interp
title('Spectrogram of Gabor Filtered SOI 1'), xlabel('Time (sec)'), ylabel('Frequency (Hz)')
set(gca,'Fontsize',12) 
colormap(hot), colorbar


%% Gabor Transform and spectrogram - window width comparison
a = 100; 

figure(3)
a_vec = [1 20 100];
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.1:L;
    vg_fft_spec = zeros(length(tslide),n);
    for j=1:length(tslide)
        g=exp(-a*(v_time_span-tslide(j)).^2); 
        vg=g.*v; 
        vg_fft=fft(vg);  
        vg_fft_spec(j,:) = fftshift(abs(vg_fft)); 
    end
    
    subplot(2,2,jj)
    pcolor(tslide,k_shift_v,vg_fft_spec.'), 
    shading interp 
    title(['Spectrogram of Gabor Filtered SOI 1 with a = ',num2str(a)],'Fontsize',12)
    set(gca,'Fontsize',12), xlabel('Time (sec)'), ylabel('Frequency (Hz)')
    colormap(hot), colorbar
end

% Plot the signal of interest's spectrogram
subplot (2,2,4);
% Time component of the signal of interest
tslide = 0:0.1:L;
v_fft_spec = repmat(fftshift(abs(v_fft)),length(tslide),1);
pcolor(tslide,k_shift_v,v_fft_spec.');
shading interp
set(gca,'Fontsize',12), xlabel('Time (sec)'), ylabel('Frequency (Hz)'), title('Spectrogram from SOI 1')
colormap(hot), colorbar

%% Oversampling and Undersampling - Tau change
a = 20; 

figure(4)
tau_over = 2;
tslide = 0:tau_over:L;
vg_fft_spec = zeros(length(tslide), length(v));
for j=1:length(tslide)
    g=exp(-a*(v_time_span-tslide(j)).^2); 
    vg=g.*v; 
    vg_fft=fft(vg);  
    vg_fft_spec(j,:) = fftshift(abs(vg_fft)); 
end
subplot(2,1,1)
pcolor(tslide,k_shift_v,vg_fft_spec.'), 
shading interp 
title(['Spectrogram of Gabor Filtered SOI 1 with tau = ',num2str(tau_over)],'Fontsize',12)
set(gca,'Fontsize',12), xlabel('Time (sec)'), ylabel('Frequency (Hz)')
colormap(hot), colorbar

tau_under = 0.1;
tslide = 0:tau_under:L;
vg_fft_spec = zeros(length(tslide), length(v));
for j=1:length(tslide)
    g=exp(-a*(v_time_span-tslide(j)).^2); 
    vg=g.*v; 
    vg_fft=fft(vg);  
    vg_fft_spec(j,:) = fftshift(abs(vg_fft)); 
end

subplot(2,1,2)
pcolor(tslide,k_shift_v,vg_fft_spec.'), 
shading interp 
title(['Spectrogram of Gabor Filtered SOI 1 with tau = ',num2str(tau_under)],'Fontsize',12)
set(gca,'Fontsize',12), xlabel('Time (sec)'), ylabel('Frequency (Hz)')
colormap(hot), colorbar

%% Mexican hat
tau = 0.1;
t_slide = 0:tau:L;
sigma = 1; %width paramater
vm_fft_spec = zeros(length(tslide), length(v));
for j=1:length(tslide)
    m_hat = (1-((v_time_span-tslide(j))/sigma).^2).*exp((-(v_time_span-tslide(j)).^2)/(2*sigma.^2));
    vm=m_hat.*v; 
    vm_fft=fft(vm);  
    vm_fft_spec(j,:) = fftshift(abs(vm_fft)); 
end

figure(5)

pcolor(tslide,k_shift_v,vg_fft_spec.'), 
shading interp 
title('Spectrogram of SOI 1 filtered with Mexican Hat Wavelet','Fontsize',12)
set(gca,'Fontsize',12), xlabel('Time (sec)'), ylabel('Frequency (Hz)')
colormap(hot), colorbar

%% PART 2
clear all; close all; clc; 

[y1,Fs] = audioread('music1.wav'); 
tr_piano=length(y1)/Fs; % record time in seconds 

[y2,Fs] = audioread('music2.wav');
tr_rec=length(y2)/Fs; % record time in seconds 

% Signal of interest 2
soi2=y1';

% Signal of interest 3
soi3=y2';

% Number of sampled points from SOI 2 
n2=length(y1);

% Number of sampled points from SOI 3
n3=length(y2);

% Time component of SOI 2
t_soi2_2 = linspace(0, tr_piano, n2+1);
t_soi2 = t_soi2_2(1:n2);

% Time component of SOI 3
t_soi3_2 = linspace(0, tr_rec, n3+1);
t_soi3 = t_soi3_2(1:n3);

% Frequency components
k2 = (1/tr_piano)*[0:n2/2-1 -n2/2:-1];
k3 = (1/tr_rec)*[0:n3/2-1 -n3/2:-1];

% Shifted freq components to be mathematically correct; scaled for Hz
% instead of angular frequencies
k2_shift = fftshift(k2);
k3_shift = fftshift(k3);
%% With Overtones
% Gabor filter
tau = 0.1;
a = 100;

tslide2 = 0:tau:tr_piano;
soi2g_fft_spec = zeros(length(tslide2), n2);
for j=1:length(tslide2)
    g=exp(-a*(t_soi2-tslide2(j)).^2); 
    soi2g=g.*soi2; 
    soi2g_fft=fft(soi2g);
    soi2g_fft_spec(j,:) = fftshift(abs(soi2g_fft)); 
end

figure(6)
subplot(2,1,1)
pcolor(tslide2,k2_shift,soi2g_fft_spec.'), 
shading interp
title('Spectrogram of Gabor Filtered SOI 2 with Overtones'), xlabel('Time (sec)'), ylabel('Frequency (Hz)')
set(gca,'Ylim', [0 1000],'Fontsize',12) 
colormap(hot), colorbar

tau = 0.1;
tslide3 = 0:tau:tr_rec;
soi3g_fft_spec = zeros(length(tslide3), n3);
for j=1:length(tslide3)
    g=exp(-a*(t_soi3-tslide3(j)).^2); 
    soi3g=g.*soi3; 
    soi3g_fft=fft(soi3g);
    soi3g_fft_spec(j,:) = fftshift(abs(soi3g_fft)); 
end
subplot(2,1,2)
pcolor(tslide3,k3_shift,soi3g_fft_spec.'), 
shading interp
title('Spectrogram of Gabor Filtered SOI 3 with Overtones'), xlabel('Time (sec)'), ylabel('Frequency (Hz)')
set(gca,'Ylim', [0 3100],'Fontsize',12) 
colormap(hot), colorbar

%% Filtering Overtones
% Gabor filter
tau = 0.1;
a = 100;

% Gaussian filter parameters
tau_gaus = 0.2;

tslide2 = 0:tau:tr_piano;
soi2g_fft_spec = zeros(length(tslide2), n2);
for j=1:length(tslide2)
    g=exp(-a*(t_soi2-tslide2(j)).^2); 
    soi2g=g.*soi2; 
    soi2g_fft=fft(soi2g);
    [maxfreq, maxindex] = max(abs(soi2g_fft(:)));
    center_freq = k2(maxindex);
    gausfilter = exp(-tau_gaus*(((k2-center_freq).^2)));
    filtered_soi2g_fft = soi2g_fft.*gausfilter;
    soi2g_fft_spec(j,:) = fftshift(abs(filtered_soi2g_fft)); 
end

figure(7)
subplot(2,1,1)
pcolor(tslide2,k2_shift,soi2g_fft_spec.'), 
shading interp
title('Spectrogram of Gabor Filtered SOI 2 with Overtones Removed'), xlabel('Time (sec)'), ylabel('Frequency (Hz)')
set(gca,'Ylim', [220 340],'Fontsize',12) 
colormap(hot), colorbar

% Gaussian filter parameters
tau_gaus = 0.01;

tslide3 = 0:tau:tr_rec;
soi3g_fft_spec = zeros(length(tslide3), n3);
for j=1:length(tslide3)
    g=exp(-a*(t_soi3-tslide3(j)).^2); 
    soi3g=g.*soi3; 
    soi3g_fft=fft(soi3g);
    [maxfreq, maxindex] = max(abs(soi3g_fft(:)));
    center_freq = k3(maxindex);
    gausfilter = exp(-tau_gaus*(((k3-center_freq).^2)));
    filtered_soi3g_fft = soi3g_fft.*gausfilter;
    soi3g_fft_spec(j,:) = fftshift(abs(filtered_soi3g_fft)); 
end
subplot(2,1,2)
pcolor(tslide3,k3_shift,soi3g_fft_spec.'), 
shading interp
title('Spectrogram of Gabor Filtered SOI 3 with Overtones Removed'), xlabel('Time (sec)'), ylabel('Frequency (Hz)')
set(gca,'Ylim', [700 1100],'Fontsize',12) 
colormap(hot), colorbar