__precompile__()

"""
A package for fitting data from auditory evidence accumulation task (Poisson clicks task) 
to evidence accumulation model.
"""
module PBupsModel

# 3rd party
import Base.convert

using MAT
using ForwardDiff
using Optim
using GeneralUtils

import ForwardDiff.DiffResults as DiffBase
# import ForwardDiff.DiffBase
# using DiffBase

export 
    
# data_handle
    LoadData,
    TrialData,
    WriteData,

# model_likelihood 
    logProbRight,
    LogLikelihood, 

# all_trials
    ComputeLL,
    ComputeLL_bbox,
    ComputeGrad,
    ComputeHess,
    ComputeHessIterative,
    TrialsLikelihood,

# optimization
    InitParams,
    GetBounds,
    ModelFitting,
    FitSummary

include("data_handle.jl")
include("model_likelihood.jl")
include("all_trials.jl")
include("model_optimization.jl")

end # module
