# Global variables
const epsilon = 10.0^(-10);
const dx = 0.25;
const dt = 0.02;
const total_rate = 40;

# === Upgrading from ForwardDiff v0.1 to v0.2
# instead of ForwardDiff.GradientNumber and ForwardDiff.HessianNumber,
# we will use ForwardDiff.Dual

convert(::Type{Float64}, x::ForwardDiff.Dual) = convert(Float64, x.value)
function convert(::Array{Float64}, x::Array{ForwardDiff.Dual})
    y = zeros(size(x));
    for i in 1:prod(size(x))
        y[i] = convert(Float64, x[i])
    end
    return y
end

struct NumericPair{X,Y} <: Number
  x::X
  y::Y
end
Base.isless(a::NumericPair, b::NumericPair) = (a.x<b.x) || (a.x==b.x && a.y<b.y)

"""
function bin_centers = make_bins(B, dx, binN)

Makes a series of points that will indicate bin centers. The first and
last points will indicate sticky bins. No "bin edges" are made-- the edge
between two bins is always implicity at the halfway point between their
corresponding centers. The center bin is always at x=0; bin spacing
(except for last and first bins) is always dx; and the position
of the first and last bins is chosen so that |B| lies exactly at the
midpoint between 1st (sticky) and 2nd (first real) bins, as well as
exactly at the midpoint between last but one (last real) and last
(sticky) bins.

Playing nice with ForwardDiff means that the *number* of bins must be predetermined.
So this function will not actually set the number of bins; what it'll do is determine their
locations. To accomplish this separation, the function uses as a third parameter binN,
which should be equal to the number of bins with bin centers > 0, as follows:
   binN = ceil(B/dx)
and then the total number of bins will be 2*binN+1, with the center one always corresponding
to position zero. Use non-differentiable types for B and dx for this to work.
"""
function make_bins(bins, B, dx, binN)
    cnt = 1
    for i=-binN:binN
        bins[cnt] = i*dx
        cnt = cnt+1
    end

    bins[end] = 2*B - (binN-1)*dx
    bins[1] = -2*B + (binN-1)*dx
end

"""
function F = Fmatrix([sigma, lambda, c], bin_centers)

Uses globals
    dt
    dx
    epsilon       (=10.0^-10)

Returns a square Markov matrix of transition probabilities.
Plays nice with ForwardDiff-- that is why bin_centers is a global vector (so that the rem
operations that go into defining the bins, which ForwardDiff doesn't know how to deal with,
stay outside of this differentiable function)

sigma  should be in (accumulator units) per (second^(1/2))
lambda should be in s^-1
c      should be in accumulator units per second
bin_centers should be a vector of the centers of all the bins. Edges will be at midpoints
       between the centers, and the first and last bin will be sticky.

dx is not used inside Fmatrix, because bin_centers specifies all we need to know.
dt *is* used inside Fmatrix, to convert sigma, lambda, and c into timestep units
"""
function Fmatrix(F::AbstractArray{T,2},params::Vector, bin_centers) where {T}
    sigma2 = params[1];
    lam   = params[2];
    c     = params[3];

    sigma2_sbin = convert(Float64, sigma2)
    # println(typeof(sigma2))
    # println(size(sigma2))
    # println(sigma2)
    # println("Converted ", sigma2_sbin)
    # println(typeof(sigma2_sbin))


    n_sbins = max(70, ceil(10*sqrt(sigma2_sbin)/dx))

    F[1,1] = 1;
    F[end,end] = 1;

    swidth = 5. * sqrt(sigma2_sbin)

    sbinsize = swidth/n_sbins #sbins[2] - sbins[1]
    
    if sbinsize > 0
        base_sbins = collect(-swidth:sbinsize:swidth)
        ps = exp.(-base_sbins.^2/(2*sigma2)) # exp(Array) -> exp.(x)
        ps = ps/sum(ps);
    else
        base_sbins = [0.]
        ps = [1.]
    end

    sbin_length = length(base_sbins)
    binN = length(bin_centers)

    mu = 0.
    for j in 2:binN-1
        if lam == 0
            mu = bin_centers[j] + c*dt#(exp(lam*dt))
        else
            mu = (bin_centers[j] + c/lam)*exp(lam*dt) - c/lam
        end

        for k in 1:sbin_length
            sbin = (k-1)*sbinsize + mu - swidth

            if sbin <= bin_centers[1] #(bin_centers[1] + bin_centers[2])/2
                F[1,j] = F[1,j] + ps[k]
            elseif bin_centers[end] <= sbin#(bin_centers[end]+bin_centers[end-1])/2 <= sbins[k]
                F[end,j] = F[end,j] + ps[k]
            else # more condition
                if (sbin > bin_centers[1] && sbin < bin_centers[2])
                    lp = 1; hp = 2;
                elseif (sbin > bin_centers[end-1] && sbin < bin_centers[end])
                    lp = binN-1; hp = binN;
                else
                    lp = floor(Int,((sbin-bin_centers[2])/dx)) + 2#find(bin_centers .<= sbins[k])[end]
                    hp = ceil(Int,((sbin-bin_centers[2])/dx)) + 2#lp+1#Int(ceil((sbins[k]-bin_centers[2])/dx) + 1);
                end

                if lp == hp
                    F[lp,j] = F[lp,j] + ps[k]
                else
                    F[hp,j] = F[hp,j] + ps[k]*(sbin - bin_centers[lp])/(bin_centers[hp] - bin_centers[lp])
                    F[lp,j] = F[lp,j] + ps[k]*(bin_centers[hp] - sbin)/(bin_centers[hp] - bin_centers[lp])
                end
            end
        end
    end
end

"""
version with inter-click interval(ici) for c_eff_net / c_eff_tot (followed the matlab code)
(which was using dt for c_eff)

function logProbRight(params::Vector)

    RightClickTimes   vector with elements indicating times of right clicks
    LeftClickTimes    vector with elements indicating times of left clicks
    Nsteps number of timesteps to simulate

Takes params
    sigma_a = params[1]; sigma_s = params[2]; sigma_i = params[3];
    lambda = params[4]; B = params[5]; bias = params[6];
    phi = params[7]; tau_phi = params[8]; lapse = params[9]

Returns the log of the probability that the agent chose Right.
"""
function logProbRight(RightClickTimes::Array{Float64,1}, LeftClickTimes::Array{Float64,1}, Nsteps::Int
    ;sigma_a = 0.01, sigma_s_R = nothing, sigma_s_L = nothing,
    sigma_i = 0.01, lambda = 0., B = 8., bias = 0.,
    phi = 1., tau_phi = 0.02, lapse_R = nothing, lapse_L = nothing,
    input_gain_weight = 0.5)

# now it considers 12p/9p function
# we can update the default parameter later

    # check whether sigma_S_L/sigma_S_R are specified
    # 1. set one of them as sigma_s
    sigma_s_default = 0.01
    if isnothing(sigma_s_R)
        if isnothing(sigma_s_L)
            sigma_s_R = sigma_s_default
            sigma_s_L = sigma_s_default
        else
            sigma_s_R = sigma_s_L
        end
    elseif isnothing(sigma_s_L)
        sigma_s_L = sigma_s_R
    end

    # same for lapse
    # check whether lapse_L/lapse_R are specified
    # 1. set one of them as lapse
    lapse_default = 0.01
    if isnothing(lapse_R)
        if isnothing(lapse_L)
            lapse_R = lapse_default
            lapse_L = lapse_default
        else
            lapse_R = lapse_L
        end
    elseif isnothing(lapse_L)
        lapse_L = lapse_R
    end

# function logProbRight(params::Vector, RightClickTimes::Array{Float64,1}, LeftClickTimes::Array{Float64,1}, Nsteps::Int)
#     sigma_a = params[1]; sigma_s_R = params[2]; sigma_s_L = params[3];
#     sigma_i = params[4]; lambda = params[5]; B = params[6]; bias = params[7];
#     phi = params[8]; tau_phi = params[9]; lapse_R = params[10]; lapse_L = params[11];
#     input_gain_weight = params[12];


    if isempty(RightClickTimes) RightClickTimes = zeros(0) end;
    if isempty(LeftClickTimes ) LeftClickTimes  = zeros(0) end;

    NClicks = zeros(Int, Nsteps);
    Lhere  = zeros(Int, length(LeftClickTimes));
    Rhere = zeros(Int, length(RightClickTimes));

    for i in eachindex(LeftClickTimes)
        Lhere[i] = ceil((LeftClickTimes[i]+epsilon)/dt)
    end
    for i in eachindex(RightClickTimes)
        Rhere[i] = ceil((RightClickTimes[i]+epsilon)/dt)
    end

    for i in Lhere
        NClicks[Int(i)] = NClicks[Int(i)]  + 1
    end
    for i in Rhere
        NClicks[Int(i)] = NClicks[Int(i)]  + 1
    end

    # === Upgrading from ForwardDiff v0.1 to v0.2
    # instead of using convert we can use floor(Int, ForwardDiff.Dual) and
    # ceil(Int, ForwardDiff.Dual)

    binN = ceil(Int, B/dx)#Int(ceil(my_B/dx))
    binBias = floor(Int, bias/dx) + binN+1
    binBias_hp = ceil(Int, bias/dx) + binN+1

    if binBias<1 binBias = 1; end
    if binBias>binN*2+1 binBias = binN*2+1; end

    if binBias_hp<1 binBias_hp = 1; end
    if binBias_hp>binN*2+1 binBias_hp = binN*2+1; end

    bin_centers = zeros(typeof(B), binN*2+1)
    make_bins(bin_centers, B, dx, binN)

    # Visualization
    a_trace = zeros(length(bin_centers), Nsteps)
    c_trace = zeros(1, Nsteps)

    ## biased lapse rate (lapse_R, lapse_L)
    a0 = zeros(typeof(sigma_a),length(bin_centers))
    a0[1] = lapse_L/2;
    a0[end] = lapse_R/2;
    a0[binN+1] = 1-lapse_L/2-lapse_R/2;

    temp_l = [NumericPair(t,-1) for t in LeftClickTimes]
    temp_r = [NumericPair(t,1) for t in RightClickTimes]
    allbups = sort!([temp_l; temp_r])

    if phi == 1
      c_eff = 1.
    else
      c_eff = 0.
    end
    cnt = 0

    Fi = zeros(typeof(sigma_i),length(bin_centers),length(bin_centers))
    Fmatrix(Fi,[sigma_i, 0., 0.0], bin_centers)
    a = Fi*a0;

    a_trace[:,1] = a;

    F0 = zeros(typeof(sigma_a),length(bin_centers),length(bin_centers))
    Fmatrix(F0,[sigma_a*dt, lambda, 0.0], bin_centers)
    for i in 2:Nsteps
        net_input = 0.
        c_eff_tot = 0.
        c_eff_net = 0.
        net_i_input = 0.
        net_c_input = 0.

        if NClicks[i-1]==0
            c_eff_tot = 0.
            c_eff_net = 0.
            a = F0*a
        else
            if cnt == 0
                c_eff = 1.
            else
                last_time = allbups[cnt].x
                ici = allbups[cnt+1].x - last_time
                c_eff = 1 + (c_eff*phi - 1) * exp(-ici/tau_phi)
            end

            c_eff_tot += c_eff * NClicks[i-1]
            for j in 1:NClicks[i-1]
                c_eff_net = c_eff_net + c_eff*allbups[cnt+j].y
            end

            net_c_input = (c_eff_tot+c_eff_net)/2 # right
            net_i_input = c_eff_tot-net_c_input # left

            # (input_gain_weight) 0 to 1 : 0-left, 1-right 
            # 0.5 is neutral 
            net_input = 2*input_gain_weight*net_c_input - 2*(1-input_gain_weight)*net_i_input

            ## biased params added
            net_sigma = sigma_a*dt + (sigma_s_R*net_i_input)/total_rate + (sigma_s_L*net_c_input)/total_rate
            F = zeros(typeof(net_sigma),length(bin_centers),length(bin_centers))
            Fmatrix(F,[net_sigma, lambda, net_input/dt], bin_centers)
            a = F*a
        end

        cnt += NClicks[i-1]
        a_trace[:,i] = a
        c_trace[i] = c_eff*exp(lambda*dt)
    end

    # likelihood of poking right
    if binBias == binBias_hp
      pright = sum(a[binBias+1:end])+a[binBias]/2
    else
      pright = sum(a[binBias+2:end]) +
      a[binBias]*((bin_centers[binBias+1] - bias)/dx/2) +
      a[binBias+1]*(0.5 + (bin_centers[binBias+1] - bias)/dx/2)
    end
    if pright-1 < epsilon && pright > 1
        pright = 1
    end
    if pright < epsilon && pright > 0
        pright = 0
    end
    
    return log(pright)
end

function LogLikelihood(RightClickTimes::Vector, LeftClickTimes::Vector, Nsteps::Int, rat_choice::Int
    ;kwargs...)
    if rat_choice > 0
        # println("Right")
        return logProbRight(RightClickTimes, LeftClickTimes, Nsteps;
            kwargs...)#make_dict(args, x)...)
    elseif rat_choice < 0
        # println("Left")
        return log(1 - exp(logProbRight(RightClickTimes, LeftClickTimes, Nsteps;
            kwargs...)))#make_dict(args, x)...)))
    end
end

"""
function (LL, LLgrad, LLhessian, bin_centers, bin_times, a_trace) =
    llikey(params, rat_choice, maxT=1, RightPulseTimes=[], LeftPulseTimes=[], dx=0.25, dt=0.02)

Computes the log likelihood according to Bing's model, and returns log likelihood, gradient, and hessian

params is a vector whose elements, in order, are
    sigma_a    square root of accumulator variance per unit time sqrt(click units^2 per second)
    sigma_s    standard deviation introduced with each click (will get scaled by click adaptation)
    sigma_i    square root of initial accumulator variance sqrt(click units^2)
    lambda     1/accumulator time constant (sec^-1). Positive means unstable, neg means stable
    B          sticky bound height (click units)
    bias       where the decision boundary lies (click units)
    phi        click adaptation/facilitation multiplication parameter
    tau_phi    time constant for recovery from click adaptation (sec)
    lapse      2*lapse fraction of trials are decided randomly

rat_choice     should be either "R" or "L"


RETURNS:


"""

# function single_trial(params::Vector, RightClickTimes::Vector, LeftClickTimes::Vector, Nsteps::Int, rat_choice::Int, hess_mode=0::Int)
#     function llikey(params::Vector)
#         logLike(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)
#     end

#     if hess_mode > 0
#         result =  HessianResult(params)
#         ForwardDiff.hessian!(result, llikey, params);
#     else
#         result =  GradientResult(params)
#         ForwardDiff.gradient!(result, llikey, params);
#     end

#     LL     = ForwardDiff.value(result)
#     LLgrad = ForwardDiff.gradient(result)

#     if hess_mode > 0
#         LLhessian = ForwardDiff.hessian(result)
#     end

#     if hess_mode > 0
#         return LL, LLgrad, LLhessian
#     else
#         return LL, LLgrad
#     end
# end

