using Distributed
using SharedArrays
using LinearAlgebra

# for using keyword_vgh()
# change the order of parameters.

# function ComputeLL(LLs::SharedArray{Float64,1}, ratdata, ntrials::Int#, args, x::Vector{T})
function ComputeLL(LLs::SharedArray, ratdata, ntrials::Int#, args, x::Vector{T})
    ;kwargs...)

    @sync @distributed for i in 1:ntrials
        RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata, i)
        Nsteps = Int(ceil(maxT/dt))

        try
            LLs[i] = LogLikelihood(RightClickTimes, LeftClickTimes, Nsteps, rat_choice ;kwargs...)
        catch e
            if isa(e, InvalidStateException) && e.state == :probRightOutsideRange
                println("Current trial: $(i)")
                println("Current parameters: $(kwargs)")
            end
            rethrow()
        end
    end

    LL = -sum(LLs)
    return LL 
end

function ComputeLL_bbox(ratdata, ntrials::Int#, args, x::Vector{T})
    ;kwargs...)
    v = kwargs[1]
    LLs = SharedArray{typeof(v)}(ntrials)#zeros(eltype(params),ntrials)
    return ComputeLL(LLs, ratdata, ntrials; kwargs...)
end

"""Compute gradient on fitparams while setting fixedparams to the given values (overriding defaults)"""
function ComputeGrad(ratdata, ntrials::Int, fitparams, fixedparams=(;))
    do_hess = false
    gradfn = (;params...) -> ComputeLL_bbox(ratdata, ntrials; params..., fixedparams...)
    LL, LLgrad = GeneralUtils.keyword_vgh(gradfn, do_hess; fitparams...)
    return LL, LLgrad
end

function ComputeGrad(ratdata, ntrials::Int, args, x::Vector{T}) where {T}
    return ComputeGrad(ratdata, ntrials, GeneralUtils.make_dict(args, x))
end

"""Compute hessian on fitparams while setting fixedparams to the given values (overriding defaults)"""
function ComputeHess(ratdata, ntrials::Int, fitparams, fixedparams=(;))
    do_hess = true
    hessfn = (;params...) -> ComputeLL_bbox(ratdata, ntrials; params..., fixedparams...)
    LL, LLgrad, LLhess = GeneralUtils.keyword_vgh(hessfn, do_hess; fitparams...)
    return LL, LLgrad, LLhess
end

function ComputeHess(ratdata, ntrials::Int, args, x::Vector{T}) where {T} 
    return ComputeHess(ratdata, ntrials, GeneralUtils.make_dict(args, x))
end

"Compute Hessian using iterative method described in Brunton et al. 2013, supplement section 3.5.1"
function ComputeHessIterative(f::Function, x0::Vector{T}; condthresh=4e15) where {T}
    ndims = length(x0)

    # Accumulated change-of-basis matrix from current subspace to original parameter space
    toOrigBasis = I

    # Found eigenvectors and eigenvalues
    Q = zeros(T, ndims, ndims)
    λ = zeros(T, ndims)
    nfilled = 0

    while nfilled < ndims
        # Get Hessian of subspace
        # projection of x0 onto eigenvectors we've computed so far
        fixedparam_x0 = Q[:, 1:nfilled] * Q[:, 1:nfilled]' * x0
        fwrapper = subspace_x -> begin
            x = toOrigBasis * subspace_x + fixedparam_x0
            return f(x)
        end

        subspace_x0 = toOrigBasis' * x0
        _, _, LLhess = GeneralUtils.vgh(fwrapper, subspace_x0)

        # Check whether we're done or need to keep going deeper
        evals, evecs = eigen(LLhess)
        if cond(LLhess) > condthresh
            # Matrix is badly scaled; save first and go deeper
            maxind = argmax(abs.(evals))
            λ[nfilled+1] = evals[maxind]
            Q[:, nfilled+1] = toOrigBasis * evecs[:, maxind]
            toOrigBasis *= evecs[:, [1:maxind-1; maxind+1:end]]
            nfilled += 1
        else
            # Fill in the rest of the eigenvalues & vectors
            Q[:, nfilled+1:end] = toOrigBasis * evecs
            λ[nfilled+1:end] = evals
            nfilled = ndims
        end
    end

    return Q * diagm(λ) * Q'
end


"Compute Hessian of DDM LL using iterative method"
function ComputeHessIterative(ratdata, ntrials::Int, args, x0::Vector{T}; kwargs...) where {T}
    f = x -> ComputeLL_bbox(ratdata, ntrials; make_dict(args, x)...)
    return ComputeHessIterative(f, x0; kwargs...)
end

"Compute Hessian of DDM LL using iterative method while setting fixedparams to the given values (overriding defaults)"
function ComputeHessIterative(ratdata, ntrials::Int, fitparams, fixedparams=(;); kwargs...)
    fitargs, x0 = GeneralUtils.to_args_format(fitparams)
    f = x -> ComputeLL_bbox(ratdata, ntrials; make_dict(fitargs, x)..., fixedparams...)
    return ComputeHessIterative(f, x0; kwargs...)
end

function TrialsLikelihood(LL::AbstractArray{T,1}, ratdata, ntrials::Int
     ;kwargs...) where {T}
    for i in 1:ntrials
        RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata, i)
        Nsteps = Int(ceil(maxT/dt))

        LL[i] = LogLikelihood(RightClickTimes, LeftClickTimes, Nsteps, rat_choice
                            ;kwargs...)
    end
end
