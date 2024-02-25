function kVector(
        StAcVector::Vector{Float64},#
        Basis::Matrix{Float64},#
        TypeOfKernel::String,#
        Dims::Vector{Int64},#
        KernelParam::Vector{Float64},#
    )

    Kernel = getfield(Main, Symbol(TypeOfKernel));
    Nb = size(Basis)[2]
    kVec = Vector{Float64}(undef, Nb)
    for i=1:Nb
        kVec[i] = Kernel(StAcVector, Basis[:,i], Dims, KernelParam)
    end

    return kVec
end


function ALD_Dist(
        StAcVector::Vector{Float64},#
        Basis::Matrix{Float64},#
        Kmat::AbstractArray{Float64,2},#
        invKmat::AbstractArray{Float64,2},#
        deltaD::Float64,#
        TypeOfKernel::String,#
        Dims::Vector{Int64},#
        KernelParam::Vector{Float64},#
    )

    Kernel = getfield(Main, Symbol(TypeOfKernel));
    Nb = size(Basis)[2]
    StAcDim = length(StAcVector)

    kVec = kVector(StAcVector, Basis, TypeOfKernel, Dims, KernelParam)

    StAcVectCoord = invKmat*kVec
    NormSqNewVector = Kernel(StAcVector, StAcVector, Dims, KernelParam)
    Dist = NormSqNewVector - kVec'*StAcVectCoord 

    if(abs(Dist)>deltaD)
        Basis = [ Basis StAcVector ]
        Kmat = [ Kmat kVec; kVec' NormSqNewVector ]
        invKmat = [ invKmat+Symmetric(StAcVectCoord*StAcVectCoord')/Dist -StAcVectCoord/Dist; -StAcVectCoord'/Dist 1/Dist ]
        StAcVectCoord = [ zeros(Nb); 1. ]
    end

    Kmat = Symmetric(Kmat)
    invKmat = Symmetric(invKmat)

    return Basis, Kmat, invKmat, StAcVectCoord
end


function ALD_Ball(
        StAcVector::Vector{Float64},#
        Basis::Matrix{Float64},#
        Kmat::AbstractArray{Float64,2},#
        invKmat::AbstractArray{Float64,2},#
        deltaB::Float64,#
        TypeOfKernel::String,#
        Dims::Vector{Int64},#
        KernelParam::Vector{Float64},#
    )

    Kernel = getfield(Main, Symbol(TypeOfKernel));
    Nb = size(Basis)[2]
    StAcDim = length(StAcVector)

    kVec = kVector(StAcVector, Basis, TypeOfKernel, Dims, KernelParam)

    StAcVectCoord = invKmat*kVec
    NormSqNewVector = Kernel(StAcVector, StAcVector, Dims, KernelParam)
    Dist = sqrt( max(0, NormSqNewVector - kVec'*StAcVectCoord) )

    if(Dist/sqrt(NormSqNewVector)>deltaB)
        Basis = [ Basis StAcVector ]
        Kmat = [ Kmat kVec; kVec' NormSqNewVector ]
        invKmat = [ invKmat+Symmetric(StAcVectCoord*StAcVectCoord')/Dist/Dist -StAcVectCoord/Dist/Dist; -StAcVectCoord'/Dist/Dist 1/Dist/Dist ]
        StAcVectCoord = [ zeros(Nb); 1. ]
    end

    Kmat = Symmetric(Kmat)
    invKmat = Symmetric(invKmat)

    return Basis, Kmat, invKmat, StAcVectCoord
end


function KRLS(
        StAcVector::Vector{Float64},#
        LossValue::Float64,#
        LossVectCoord::Vector{Float64},#
        Basis::Matrix{Float64},#
        Kmat::AbstractArray{Float64,2},#
        invKmat::AbstractArray{Float64,2},#
        delta::Float64,#
        TypeOfKernel::String,#
        Dims::Vector{Float64},#
        KernelParam::Vector{Float64},#
    )

    Kernel = getfield(Main, Symbol(TypeOfKernel));
    Nb = size(Basis)[2]
    StAcDim = length(StAcVector)

    kVec = kVector(StAcVector, Basis, TypeOfKernel, Dims, KernelParam)
end
