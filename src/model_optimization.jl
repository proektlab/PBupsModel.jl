# updating for generalizing the model. 
# user will give args to init parameters 
# use 'dictionary' -> each parameter has different range

# utilize make_dict and defaults.

const ParamDict = Dict{String, Float64}

function InitParams(args, seed_mode=1)

    # Parameters (match the parameter order with original code)
    # --> order doesn't matter now, it will be matched with the order of parameters

    x_val = zeros(length(args))

    # random seed    
    if seed_mode == 1
        rseed_param = Dict("sigma_a" => rand()*50., 
        "sigma_s_R" => rand()*50.,
        "sigma_s_L" => rand()*50.,
        "sigma_i" => rand()*30.,
        "lambda" => randn()*.01,
        "B" => rand()*12. + 5.,
        "bias" => randn(),
        "phi" => rand()*0.99+0.01,
        "tau_phi" => 0.695*rand()+0.005,
        "lapse_R" => rand()*.5,
        "lapse_L" => rand()*.5,
        "input_gain_weight" => rand())
        rseed_param["bias_rel"] = rseed_param["bias"] / rseed_param["B"]

        for i in eachindex(args)
            x_val[i] = rseed_param[args[i]]
        end
    # bing's rat avg parameter set
    elseif seed_mode == 2
        params = [-0.2767, 2.0767, 75.9600, 1.9916, 8.9474, 0.1694, 0.0964, -0.0269, 0.0613]
        defaults_param = Dict("sigma_a" => 2.0767, 
        "sigma_s_R" => 75.9600,
        "sigma_s_L" => 75.9600,
        "sigma_i" => 1.9916,
        "lambda" => -0.2767,
        "B" => 8.9474,
        "bias" => -0.0269,
        "phi" => 0.1694,
        "tau_phi" => 0.0964,
        "lapse_R" => 0.0613,
        "lapse_L" => 0.0613,
        "input_gain_weight" => 0.5)
        defaults_param["bias_rel"] = defaults_param["bias"] / defaults_param["B"]

        for i in eachindex(args)
            x_val[i] = defaults_param[args[i]]
        end

    # simple fixed parameter set    
    elseif seed_mode == 3
        simple_param = Dict("sigma_a" => 1., 
        "sigma_s_R" => 0.1,
        "sigma_s_L" => 0.1,
        "sigma_i" => 0.2,
        "lambda" => -0.0005,
        "B" => 6.1,
        "bias" => 0.1,
        "phi" => 0.3,
        "tau_phi" => 0.1,
        "lapse_R" => 0.1,
        "lapse_L" => 0.1,
        "input_gain_weight" => 0.5)
        simple_param["bias_rel"] = simple_param["bias"] / simple_param["B"]

        for i in eachindex(args)
            x_val[i] = simple_param[args[i]]
        end
    end

    return x_val   
end

function GetBounds(args;
    lb_overrides::ParamDict = ParamDict(), ub_overrides::ParamDict = ParamDict())
    l_b = Dict("sigma_a" => 0., 
        "sigma_s_R" => 0.,
        "sigma_s_L" => 0.,
        "sigma_i" => 0.,
        "lambda" => -5.,
        "B" => 5.,
        "bias" => -5.,
        "bias_rel" => -1.,
        "phi" => 0.01,
        "tau_phi" => 0.005,
        "lapse_R" => 0.,
        "lapse_L" => 0.,
        "input_gain_weight" => 0.)
    merge!(l_b, lb_overrides)

    u_b = Dict("sigma_a" => 200., 
        "sigma_s_R" => 200.,
        "sigma_s_L" => 200.,
        "sigma_i" => 40.,
        "lambda" => 5.,
        "B" => 25.,
        "bias" => 5.,
        "bias_rel" => 1.,
        "phi" => 1.2,
        "tau_phi" => 1.5,
        "lapse_R" => 1.,
        "lapse_L" => 1.,
        "input_gain_weight" => 1.)
    merge!(u_b, ub_overrides)

    l = zeros(length(args))
    u = zeros(length(args))

    for i in eachindex(args)
        l[i] = l_b[args[i]]
        u[i] = u_b[args[i]] 
    end

    return l, u
end


function ModelFitting(args, x_init::Vector, ratdata, ntrials; kwargs...)
    return ModelFitting(make_dict(args, x_init), ratdata, ntrials; kwargs...)
end

"""
Fit parameters in fitparams (dict or named tuple), holding parameters in fixedparams constant.
Currently algorithm can be :lbfgs or :ipnewton, or :lbfgs_nlopt to use the NLopt package instead of Optim.jl.
iterative_hessian can be set to true to use a more precise method for computing the saved Hessian
(this can also be done after the fact). If using interior-point newton algorithm, the non-iterative
Hessian is always used for fitting.
"""
function ModelFitting(fitparams, ratdata, ntrials;
        fixedparams::NamedTuple = (;), iterative_hessian=false, algorithm=:lbfgs, algo_params=(;),
        optim_overrides=(;), lb_overrides::ParamDict = ParamDict(), ub_overrides::ParamDict = ParamDict())
    fitargs, x_init = GeneralUtils.to_args_format(fitparams)
    println("Fitting parameters: $(fitargs)")
    l, u = GetBounds(fitargs; lb_overrides=lb_overrides, ub_overrides=ub_overrides)

    function LL_f(x::Vector)
        return ComputeLL_bbox(ratdata["rawdata"], ntrials; make_dict(fitargs, x)..., fixedparams...)
    end

    # # updated for julia v0.6 (in-place order)
    # function LL_g!(grads::Vector{T}, x::Vector{T}) where {T}
    #     _, LLgrad = ComputeGrad(ratdata["rawdata"], ntrials, make_dict(fitargs, x), fixedparams)
    #     grads[:] = LLgrad
    # end

    function LL_f(x::Vector, grads::Vector)
        if length(grads) > 0
            LL, LLgrad = ComputeGrad(ratdata["rawdata"], ntrials, make_dict(fitargs, x), fixedparams)
            grads[:] = LLgrad
        else
            LL = LL_f(x)
        end
        return LL
    end

    # function LL_h!(hess::Matrix{T}, x::Vector{T}) where {T}
    #     _, _, LLhess = ComputeHess(ratdata["rawdata"], ntrials, make_dict(fitargs, x), fixedparams)
    #     hess[:] = LLhess
    # end
    
    # function my_line_search!(df, x, s, x_scratch, gr_scratch, lsr, alpha,
    #     mayterminate, c1::Real = 1e-4, rhohi::Real = 0.5, rholo::Real = 0.1, iterations::Integer = 1_000)
    #     initial_alpha = 0.5
    #     LineSearches.bt2!(df, x, s,x_scratch, gr_scratch, lsr, initial_alpha,
    #                   mayterminate, c1, rhohi, rholo, iterations)
    # end

    # d4 = OnceDifferentiable(LL_f,LL_g!,x_init)
    # d4 = TwiceDifferentiable(LL_f, LL_g!, LL_h!, x_init)
    opts = (;
        g_abstol=1e-12, outer_g_abstol=1e-12,
        x_abstol=1e-10, outer_x_abstol=1e-10,
        f_reltol=1e-6, outer_f_reltol=1e-6,
        iterations=10,  # inner
        store_trace=true,
        show_trace=true,
        extended_trace=true,
        optim_overrides...)

    history, fit_time = optimize_ddm(Val(algorithm), LL_f, l, u, x_init, opts, algo_params)
    println(history.minimizer)
    println(history)

    ## do a single functional evaluation at best fit parameters and save likely for each trial
    # likely_all = zeros(typeof(sigma_i),ntrials)
    x_bf = history.minimizer #.minimum
    # Likely_all_trials(likely_all, x_bf, ratdata["rawdata"], ntrials)

    if iterative_hessian
        LLhess = ComputeHessIterative(ratdata["rawdata"], ntrials, make_dict(fitargs, x_bf), fixedparams)
    else
        _, _, LLhess = ComputeHess(ratdata["rawdata"], ntrials, make_dict(fitargs, x_bf), fixedparams)
    end

    if isa(history, Optim.OptimizationResults)
        Gs = zeros(length(history.trace),length(x_init))
        Xs = zeros(length(history.trace),length(x_init))
        fs = zeros(length(history.trace))

        for i=1:length(history.trace)
            tt = getfield(history.trace[i],:metadata)
            fs[i] = getfield(history.trace[i],:value)
            Gs[i,:] = tt["g(x)"]
            Xs[i,:] = tt["x"]
        end
    else  # NLopt doesn't give us history information :(
        empty = Matrix{Float64}(undef, 0, 0)
        Gs = empty
        Xs = empty
        fs = empty
    end

    D = Dict([("x_init",x_init),    
                ("parameters",fitargs),
                ("trials",ntrials),
                ("f",history.minimum), 
                ("x_converged",history.x_converged),
                ("f_converged",history.f_converged),
                ("g_converged",history.g_converged),                            
                ("grad_trace",Gs),
                ("f_trace",fs),
                ("x_trace",Xs),                         
                ("fit_time",fit_time),
                ("x_bf",history.minimizer),
                ("myfval", history.minimum),
                ("hessian", LLhess)
                ])

    # saveto_filename = *("julia_out_",ratname,"_rseed.mat")
    # WriteFile(mpath, filename, D)
    # matwrite(saveto_filename, Dict([("x_init",params),
    #                                 ("trials",ntrials),
    #                                 ("f",history.minimum), 
    #                                 ("x_converged",history.x_converged),
    #                                 ("f_converged",history.f_converged),
    #                                 ("g_converged",history.g_converged),                            ("grad_trace",Gs),
    #                                 ("f_trace",fs),
    #                                 ("x_trace",Xs),                         
    #                                 ("fit_time",fit_time),
    #                                 ("x_bf",history.minimizer),
    #                                 ("myfval", history.minimum),
    #                                 ("hessian", LLhess)
    #                                 ]))
    return D
end

function optimize_ddm(::Val{:lbfgs}, LL_f::Function, l, u, x_init, opts::NamedTuple, algo_params::NamedTuple)
    problem = OnceDifferentiable(LL_f, x_init, autodiff=:forward)
    fit_info = @timed Optim.optimize(problem, l, u, x_init, Fminbox(LBFGS(;algo_params...)), Optim.Options(;opts...))
    history = fit_info.value
    fit_time = fit_info.time
    return history, fit_time
end

function optimize_ddm(::Val{:ipnewton}, LL_f::Function, l, u, x_init, opts::NamedTuple, algo_params::NamedTuple)
    # remove iterations b/c it is only supposed to be for the inner optimizer, which this method doesn't use
    opts = Base.structdiff(opts, (;iterations=nothing))
    if haskey(opts, :outer_iterations)
        opts = (;opts..., iterations=opts.outer_iterations)
    end
    problem = TwiceDifferentiable(LL_f, x_init, autodiff=:forward)
    constraints = TwiceDifferentiableConstraints(l, u)
    fit_info = @timed Optim.optimize(problem, constraints, x_init, IPNewton(;algo_params...), Optim.Options(;opts...))
    history = fit_info.value
    fit_time = fit_info.time
    return history, fit_time
end

function optimize_ddm(::Val{:lbfgs_nlopt}, LL_f::Function, l, u, x_init, opts::NamedTuple, algo_params::NamedTuple)
    opt = Opt(:LD_LBFGS, length(x_init))
    opt.lower_bounds = l
    opt.upper_bounds = u
    opt.min_objective = LL_f

    # translate options from Optim.jl
    haskey(opts, :f_tol) && (opt.ftol_rel = opts.f_tol)
    haskey(opts, :f_reltol) && (opt.ftol_rel = opts.f_reltol)
    haskey(opts, :f_abstol) && (opt.ftol_abs = opts.f_abstol)

    haskey(opts, :x_tol) && (opt.xtol_abs = opts.x_tol)
    haskey(opts, :x_reltol) && (opt.xtol_rel = opts.x_reltol)
    haskey(opts, :x_abstol) && (opt.xtol_abs = opts.x_abstol)

    haskey(opts, :time_limit) && (opt.maxtime = opts.time_limit)

    if haskey(opts, :f_calls_limit)
        opt.maxeval = opts.f_calls_limit
    elseif haskey(opts, :g_calls_limit)
        opt.maxeval = opts.g_calls_limit
    end

    fit_info = @timed NLopt.optimize(opt, x_init)
    minf, minx, ret = fit_info.value
    fit_time = fit_info.time

    res = (minimizer=minx, minimum=minf,
        x_converged=(ret == :XTOL_REACHED),
        f_converged=(ret == :FTOL_REACHED),
        g_converged=(ret == :SUCCESS)  # probably not really what this means
    )
    return res, fit_time
end

function FitSummary(mpath, filename, D)
    WriteFile(mpath, filename, D)
end

## matwrite -> data_handle?