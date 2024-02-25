using LinearAlgebra
using Distributed
using Random
using Plots
using Printf
using Kronecker
using JLD2

include("./InvertedPendulum.jl")
include("./Kernels.jl")
include("./UpdateBasis.jl")


# Algorithm Params
deltaD = .001
KernelType = "Gauss"
global GaussKernelParam = [2.; 0.]

Kernel = getfield(Main, Symbol(KernelType))
KernelParam = getfield(Main, Symbol(KernelType*"KernelParam"))

# Simulation Params
NoExp = 3
NoIter = 20
#NoScanSteps = 120
#NoScanEps = 50
NoSteps = 250
NoEps = 100
SimSteps = 200

Discount = .99


AvAccLoss = zeros(NoIter)

for Exp=1:NoExp

    AccLoss = zeros(NoIter)
    Na = length(torques)
    StDim = 2
    StAcDim = 10 # as StAcVec = [StVec, torques[actionIndx]/max(abs(torques))] 
    Dims = [StAcDim] # StAcDim

    States = Matrix{Float64}(undef, StDim, 0)
    Actions = Vector{Int64}(undef, 0)
    Losses = Vector{Float64}(undef, 0)
    NextStates = Matrix{Float64}(undef, StDim, 0)
    Absorbs = Vector{Int64}(undef, 0)

    # Scanning
    for Eps=1:NoEps
        s = [pi*(rand()*2-1), 0.]
        for steps=1:NoSteps
            States = [States s]
            a = randperm(Na)[1]
            Actions = [Actions; a]

            s_nxt, loss, absorb = InvertedPendulumStep(s, a, EnvParams)

            Losses = [Losses; loss]
            NextStates = [NextStates s_nxt]
            Absorbs = [Absorbs; absorb]

            s = s_nxt

            if absorb==1
                break
            end
        end #for NoSteps
    end #for NoEps

    T = length(Actions)
    InitStAcVec = StateAction(States[:,1], Actions[1], EnvParams)

    Basis = Matrix{Float64}(undef, StAcDim, 1)
    Basis[:,1] = InitStAcVec

    Kmat = Matrix{Float64}(undef, 1, 1)
    Kmat[1,1] = Kernel(Basis[:,1], InitStAcVec, Dims, KernelParam)

    invKmat = Matrix{Float64}(undef, 1, 1)
    invKmat[1,1] = 1/Kmat[1,1]

    # creating Dictionary via ALD
    for t=1:T
        z = StateAction(States[:,t], Actions[t], EnvParams)
        Basis, Kmat, invKmat, dump = ALD_Dist(z,#
                                              Basis,#
                                              Kmat,#
                                              invKmat,#
                                              deltaD,#
                                              KernelType,#
                                              Dims,#
                                              KernelParam,#
                                             )
    end #for Tscan

    Nb = size(Basis)[2]
    @printf "Test %i/%i: #basis vecs %i\n" Exp NoExp Nb

    QVectCoord = zeros(Nb)

    #KLSPI
    for Iter=1:NoIter

        # KLSTD-Q 
        MatA = zeros(Nb, Nb)
        VecB = zeros(Nb)
        for t=1:T
            z = StateAction(States[:,t], Actions[t], EnvParams)
            zVec = kVector(z, Basis, KernelType, Dims, KernelParam)

            s_nxt = NextStates[:,t]
            Qvalues_nxt = Vector{Float64}(undef, Na)
            for aIndx=1:Na
                NxtStAcVec = StateAction(s_nxt, aIndx, EnvParams)
                kVec_nxt = kVector(NxtStAcVec, Basis, KernelType, Dims, KernelParam)
                Qvalues_nxt[aIndx] = kVec_nxt'*QVectCoord
            end #for Na

            a_nxt = argmin(Qvalues_nxt)
            znext = StateAction(s_nxt, a_nxt, EnvParams)
            zVec_nxt = kVector(znext, Basis, KernelType, Dims, KernelParam)

            MatA = MatA + zVec*(zVec' - Discount*(1-Absorbs[t])*zVec_nxt')
            VecB = VecB + zVec*Losses[t]
        end

        QVectCoord = pinv(MatA)*VecB

        # simulate
        s = [pi; 0.]
        for _ in 1:SimSteps
            Qvalues = Vector{Float64}(undef, Na)
            for aIndx=1:Na
                StAcVec = StateAction(s, aIndx, EnvParams)
                kVec = kVector(StAcVec, Basis, KernelType, Dims, KernelParam)
                Qvalues[aIndx] = kVec'*QVectCoord
            end #Na

            a = argmin(Qvalues)

            s_nxt, loss, absorb = InvertedPendulumStep(s, a, EnvParams)

            s = s_nxt

            AccLoss[Iter] += loss
            if absorb==1
                break
            end
        end #SimSteps

        if (Iter%1==0)
            @printf "\tIter(%i) AccLoss %.2f\n" Iter AccLoss[Iter]
        end
    end #for NoIter

    global AvAccLoss = AvAccLoss*(Exp-1)/Exp + AccLoss/Exp

end #for NoExp


plot(1:NoIter,#
     AvAccLoss,#
     legend=:outertopright,#
     label="KLSPI",#
     ylims=(0, SimSteps+1),#
     dpi = 600,#
    )

filename = "./KLSPI-2.jld2"
Result = Dict("AvAccLoss"=>AvAccLoss)

#@save filename Result

figname = "./KLSPI-2.png"
savefig(figname)
