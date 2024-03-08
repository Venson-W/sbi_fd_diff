%% README
% This code contains the implementation fo the SBI-FD method that is described in
% Yuhan, W., Heimisson, E. R. (2024). A coupled finite difference-spectral boundary integral 
% method with applications to fluid diffusion in fault structures. International Journal for 
% Numerical and Analytical Methods in Geomechanics.
% If code is used please reference the above study appropriately.

% In this implementation, we establish an injection scenario specifically designed for 
% a geological fault structure surrounded by host rocks. The model's permeability 
% architecture is symmetrically distributed. Injection begins at the domain's central point. 
% The code is structured into three main components: 
% (1) Model pre-processing, where we define the model's geometry, initialize parameters, 
% and generate wavenumber series to compute convolution kernels; 
% (2) Convolution kernel pre-allocation and downsizing, where we truncate and downsample the convolution kernels, 
% as elaborated in Section 4.2 of the NAG paper; 
% (3) Time-stepping loop, where the pressure field is updated using a hybrid computational scheme.

% Hybrid SBI-FD method
clear; clc; close all;                       % clean history

% Model pre-processing
% The configuration of the fault geometry and discretization
L = 50;                                      % The length of fault, Unit: m
spatial_power = 7;                           % The exponent to determine grid number 
nx = 2^spatial_power;                        % Counts of the x-axis node
W_fz = 1.5;                                  % Width of fault zone, Unit: m
W_fc = 0.15;                                 % Width of fault core, Unit: m
Boundary_S = L/4 - W_fz/2;                   % The upper boundary, Unit: m 
Boundary_N = L/4 + W_fz/2;                   % The lower boundary, Unit: m 
xx = linspace(-L/2 , L/2, nx);               % X-axis node coordination
dx = L/(nx - 1);                             % X-axis grid size, Unit: m
dy = dx/16;                                  % Y-axis grid size, Unit: m
ny = round((Boundary_N - Boundary_S)/dy) + 1;% Counts of the y-axis node
yy = linspace(Boundary_S, Boundary_N, ny);   % Y-axis node coordination

% Permeability architecture of the fault structure
% Fault zone mobility 
M_kappa_x =  1.4e-12.* ones(nx+1, ny+1);                                           % Damage zone 
M_kappa_x(:, round((W_fz/2 - W_fc/2)/dy+1):round((W_fz/2 + W_fc/2)/dy+1)) = 5e-14; % Fault core
M_kappa_y =  8e-13 .* ones(nx+1, ny+1);                                            % Damage zone
M_kappa_y(:, round((W_fz/2 - W_fc/2)/dy+1):round((W_fz/2 + W_fc/2)/dy)+1) = 5e-15; % Fault core
kappa_hostrock = 1.0e-15;                                                          % Host rock
% Dual node layers of host rock to ascertain stability
M_kappa_y(:, 1:2) = kappa_hostrock;
M_kappa_y(:, end-1:end) = kappa_hostrock;
M_kappa_x(:, 1:2) = kappa_hostrock;
M_kappa_x(:, end-1:end) = kappa_hostrock;
% Compressibility
beta = 1.0e-10;

% Timestep - Courant–Friedrichs–Lewy condition for controlling 
sigma = 0.2;                                                           % Relaxation factor
dt = sigma * min(dx, dy)^2 / (max(M_kappa_x,[],'all') / beta);         % Time step, Unit: s
t_end = 1*3600;                                                        % The ending time for simulation, Unit: s   
nt = round(t_end / dt);                                                % The number of time step for simulations

% Injection configuration
% injection rate control
t_inject = 0.5*3600;                                                   % The ending time for injection, Unit: s
fi = 5e-7;                                                             % Injection rate, Unit: m/s
inject = @(t, beta) fi*t/(beta*dx*dy)*(heaviside(-t + t_inject))...    % The heaviside function to control the stop
         +  heaviside( t - t_inject )*fi*t_inject/(beta*dx*dy) ;       % of injection, t_injection and beta will be input
nt_inject = round(t_inject/dt);                                        % The number of time step going for injection
% Injection point allocation
Domain_mid = (Boundary_N - Boundary_S)/2 - W_fc/2;
inject_x = [round((nx+1)/3) round((nx+1)/2) round((nx+1)*2/3)];
inject_y = [round((ny + 1)/4) round((ny + 1)/2) round((ny + 1)/5)];

% Pre-processing for the SBIM
% 1. Compute the period of the length
P = xx(end) - xx(1);    
% 2. Generate the wavenumber domain vector 'k' using fftshift to center the zero frequency component.
% The vector ranges from -nx/2 to nx/2-1, scaled by the period P.
k = fftshift(2 * pi * (-nx/2:1:(nx/2 - 1)) ./ P);                      
% 3. Set the first element (k=0) of wavenumber to a small value adjust for low-frequency offset
k(1) = 2*pi/(10*L);

% Pore pressure initialization
p = zeros(nx, ny);               % Initial p matrix
pn = zeros(nx, ny);              % The p in the last timestep for updating
i = 2:nx - 1;                    % Define the interior grid points along the x-axis, excluding boundary points
j = 2:ny - 1;                    % Define the interior grid points along the y-axis, excluding boundary points

% Boundary Condition
% PnS and PnN are solved by the SBIM and thereby not given here
PnW=0;                           % x=0 Direchlet B.C.
PnE=0;                           % x=L Direchlet B.C.

% Initialization of convolution kernel (Pre-allocation)
% Cell array data structure is used somewhere for saving storage
% 
M_kernel = cell(1, nx);
ts_index = zeros(1,length(k));
lr_index = zeros(1,length(k));
M_kernelTV = zeros(nt,1);
% Parameter for storing down
M_kernelR = cell(1, nx);
intind = cell(1,nx);
indr = cell(1,nx);

% Tolerance used in controlling approximation (i.e. truncation, downsampling and convolution)
t_tol = 1e-4;                   % Tolerance for identifying the point since which the kernel vary trivially
int_tol = 1.0e-4;               % Tolerance for restraining the downsampling induced error
tol_up = 1.0e-3;                % Tolerance for identifying the frequency updating convolution

% Set the downsampling up to 2*maxn+1
% Too large maxn brings up unstable/dispersive simulation results
% Under this case, maxn = 10, 5 and 0 works
% The proper value for maxn depends on kappa_hostrock and sigma
maxn = 10;

disp('Start Pre-allocating Convolution') 

% Convolution kernel downsizing
% Truncation: First, iterate the whole time series for each node along the fault length 
for i_xx = 1:nx
   for i_tt = 1:nt
        % Computing the convolution kernel with half-dt for higher precision
        M_kernelT = sqrt(kappa_hostrock/beta)*exp(-kappa_hostrock/beta * k(i_xx) ^ 2 * (i_tt*dt - dt/2)) / (sqrt(pi * (i_tt*dt - dt/2)));        
        % Compute the kernel value for all x coordinates at the first time step
        if i_tt == 1       
            M_kernelTV(i_tt,1) = M_kernelT;
            ts_index(1,i_xx) = i_tt;
        % If the tolerance reached, update the truncating point and store the updated kernel 
        elseif M_kernelT / M_kernelTV(1) > t_tol
            ts_index(1,i_xx) = i_tt;
            M_kernelTV(i_tt,1) = M_kernelT;    
        else
            ts_index(1,i_xx) = i_tt;
            M_kernel{i_xx}(i_tt,1) = M_kernelT;    
           continue 
        end
   end
   M_kernel{i_xx}(:,1) = M_kernelTV(1:i_tt,1);
end

disp('Truncation Done') 

% Downsampling - redue the size of kernel matrix
for i_xx = 1:nx
           % Refence array used for evaluating the error induced by downsampling 
           ref = cumsum(M_kernel{i_xx}(:,1));
           % The temporal time step count during downsampling
           ishift = 1;
           % The count of inddex after downsampling
           subind = ishift;
           % Temporal maxn
           maxnT = maxn;
           % Decide if the approximating range being too large, if maxnT = 1, 
           % no reducing action will be made
            if ishift + (2*maxn+1) > ts_index(i_xx)
                disp('maxn is too big for this k, setting to 1')
                maxnT = 1;
            end
           startnT = 1;
           while ishift + (2*maxnT+1) <= ts_index(i_xx)
                if startnT < maxnT
                    for n = startnT:maxnT
                        subindT = [subind;ishift+2*n+1];
                        testdiffT = diff([0;subindT]);
                        inddexT = subindT - (testdiffT - 1)/2;
                        interr = abs(sum(testdiffT.*M_kernel{i_xx}(inddexT,1)) ...
                             - ref(2*n+1 + ishift))./ref(2*n+1 + ishift); 
                        if n == maxnT && interr < 0.25*int_tol
                            interr = 0; % The error is sufficiently small to be neglected
                        end
                        if interr > int_tol 
                            break       % The tolerance reached, such that jump out the approximation cycle
                        end
                    end
                    n = n - 1;
                    subind = [subind;ishift + 2*n+1];
                    % Recalculate testdiffT and inddexT for the final adjustment
                    testdiffT = diff([0;subind]);
                    inddexT = subind - (testdiffT - 1)/2; 
                    ishift = ishift + (2*n+1) ;
                    % Adjust maxnT based on remaining indices
                    if ishift + (2*maxnT+1) > ts_index(i_xx)
                       maxnT = (- ishift + ts_index(i_xx) - 1)/2;
                       if maxnT < 1
                           break
                       end
                    end
                    startnT = n; 
                    % Final adjustments based on error and index count
                    if interr == 0 && startnT + 1 == maxnT
                       startnT = maxnT; 
                    end
                    if startnT > maxnT
                        startnT = floor(maxnT);
                    end
                else
                    % Finalize downsampling if startnT is not less than maxnT
                    n = maxnT;
                    % Calculate final indices based on remaining steps
                    nf = floor(((ts_index(i_xx) - ishift)   -  1)/(2*n));  
                    nv = (n*ones(nf,1));
                    subind = [subind; ishift + cumsum(2*nv+1) ];
                    subind(subind > ts_index(i_xx)) = [];
                    testdiffT = diff([0;subind]); 
                    inddexT = subind - (testdiffT - 1)/2; 
                    break
                end
           end
     
           % Final assignment of downsampling results
           indr{i_xx}(:,1) = flip(inddexT);  
           intind{i_xx}(:,1) = testdiffT;      
           M_kernelR{i_xx}(:,1) = flip(intind{i_xx}(:,1).*M_kernel{i_xx}(inddexT,1));
           % Record the length of the final index array for each kernel
           lr_index(1,i_xx) = length(inddexT);
end

disp('Compression Done') 

% The reduced kernel matrices for northern and southern boundaries, respectively
M_kernel_N = M_kernelR;
M_kernel_S = M_kernelR;
% Purge the memory for these two matrices
clearvars M_kernel                
clearvars M_kernelR

% Store the length information 
maxM = zeros(size(lr_index));
for i_xx = 1:nx
maxM(i_xx) = length(M_kernel_N{i_xx}(:, 1));
end

jn_hat_storage_NM = zeros(nx, nt);
jn_hat_storage_SM = zeros(nx, nt); 
M_conv_S = zeros(size(k));
M_conv_N = zeros(size(k));
ind_up = zeros(size(k)) + 1;
i_xv = 1:length(k);

% Wanna to make a movies from the list of data file? Uncomment these code
% aviobj=VideoWriter([path '/data/width6m/Uneven_Injection0d5h_Diffusion2h'],'MPEG-4');
% aviobj.FrameRate = 100; open(aviobj);
% Setting screen size & color


% For creating animation
[X,Y]  = meshgrid(xx,yy);

disp('Time Stepping Begins') 

% Time-stepping loop
for i_t = 1:nt    
    % Animation part. Please uncomment for a faster computation
    % Update the visualizing window every 500 time steps
    % The color scale demonsrates the pore pressure field p, Unit: MPa
    if rem(i_t,500)==0        
        p_field = surf(X, Y, p'/1000000);  
        grid off %plotting the field variable   
        view(2)
        xlabel('Spatial coordinate (x) \rightarrow')
        ylabel('{\rightarrow} Spatial coordinate (y)')
        box on
        clim([0 5]);
        colormap(flipud(hot));
        axis ([-L/2 L/2 0 L/2])
        daspect([1 1 1])
        hcol = colorbar;
        ylabel(hcol,'Pressure [MPa]')
        x = [-L/2 -L/2; L/2 L/2; L/2 L/2; -L/2 -L/2];
        y = [(Boundary_N) 0; (Boundary_N) 0; L/2 Boundary_S; L/2 Boundary_S];    
        xc = [-L/2 ; L/2 ; L/2 ; -L/2 ];
        yc = [(L/4 + W_fc/2); (L/4 + W_fc/2) ; L/4 - W_fc/2; L/4 - W_fc/2];
        shading flat
        patch(x,y,[.7 .7 .7],'EdgeColor','none', 'FaceAlpha', 0.25);
        colorbar;
        title({['2-D Injection (SBI-FDM)'];['time (\itt) = ',num2str(i_t*dt/3600),' h']})
        drawnow;
    end
    % FDM part
    pn = p;
    % Update the pressure field with the staggering scheme 
    p(i,j) = pn(i,j) + (1/beta)*dt*(M_kappa_x(i+1,j).*(pn(i+1,j)-pn(i,j)) - M_kappa_x(i,j).*(pn(i,j)-pn(i-1,j)))/dx^2 + ...
             (1/beta)*dt*(M_kappa_y(i,j+1).*(pn(i,j+1)-pn(i,j)) - M_kappa_y(i,j).*(pn(i,j)-pn(i,j-1)))/dy^2;
    % Injection source update
    if i_t*dt < t_inject
        for i_inj = 1:length(inject_x)
            p(inject_x(i_inj), inject_y(i_inj)) = p(inject_x(i_inj), inject_y(i_inj))+ inject(i_t*dt, beta) - inject((i_t-1)*dt, beta);
        end  
    end
    % Calculate flux jn in the coupled interface
    jn_N = -M_kappa_y(2:end,end-1) .* (p(:,end) - p(:,end-1))/dy;
    jn_S = -M_kappa_y(2:end,2).* (p(:,1) - p(:,2))/dy;
    % SBI part
    if i_t>1
    jn_hat_preN = jn_hat_storage_NM(:, i_t-1);
    jn_hat_preS = jn_hat_storage_SM(:, i_t-1);
    else
    jn_hat_preN = 0;
    jn_hat_preS = 0;   
    end
    % Storaging the flux history
    jn_hat_storage_NM(:, i_t) = 0.5*(fft(jn_N) + jn_hat_preN);
    jn_hat_storage_SM(:, i_t) = 0.5*(fft(jn_S) + jn_hat_preS);
   
    % Identify the x coordinates where the convolution updates have significantly lagged, 
    % based on a tolerance threshold.
    Iup = (i_t - ind_up)./ts_index > tol_up;                    
    i_xred = i_xv(Iup);                                         
    % Loop for computing convolution 
    for i_x = i_xred    
    % Choose the way for computing convolution based on if the current time index i_t is less 
    % than the truncating threshold at this x coordinate point (i_x)
            if i_t < ts_index(1,i_x)
                indvT = flip(indr{i_x}(:,1));
                Inonneg = -indvT + i_t > 0;
                indvT = indvT(Inonneg);
                M_conv_N(i_x) = jn_hat_storage_NM(i_x, indvT) * ((M_kernel_N{i_x}((maxM(i_x) - length(indvT)+1):maxM(i_x))));
                M_conv_S(i_x) = jn_hat_storage_SM(i_x, indvT) * ((M_kernel_S{i_x}((maxM(i_x) - length(indvT)+1):maxM(i_x))));
            else
                indvT = i_t  -  indr{i_x}(:,1) + 1 ; % pick up the nodes 
                M_conv_S(i_x) = jn_hat_storage_SM(i_x,indvT) * ( M_kernel_S{i_x}  );
                M_conv_N(i_x) = jn_hat_storage_NM(i_x,indvT) * ( M_kernel_N{i_x}  );
            end
    % Record the current time step when the convolution is updated
            ind_up(i_x) = i_t;
    end
    % SBI boundary - Direchlet
    p(:,end) = dt/kappa_hostrock*real(ifft(M_conv_N));
    p(:,1) =   dt/kappa_hostrock*real(ifft(M_conv_S));
    % FD boundary - Direchlet
    p(1,:)= PnW;
    p(nx,:)= PnE;
end

disp('Time Stepping Done')

