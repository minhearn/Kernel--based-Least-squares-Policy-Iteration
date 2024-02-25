function Poly(
        StAcVector1::Vector{Float64},#
        StAcVector2::Vector{Float64},#
        Dims::Vector{Int64},#
        KernelParam::Vector{Float64},#
    )
    
    K = (StAcVector1'*StAcVector2 + KernelParam[2])^(KernelParam[1])

    return K
end


function Gauss(
        StAcVector1::Vector{Float64},#
        StAcVector2::Vector{Float64},#
        Dims::Vector{Int64},#
        KernelParam::Vector{Float64},#
    )

    St1 = StAcVector1[ 1: Dims[1] ]
    St2 = StAcVector2[ 1: Dims[1] ]
    K = exp( -((norm(St1-St2))^2)/2/KernelParam[1] )

    for i=1:length(Dims)-1
        St1 = StAcVector1[ sum(Dims[1:i])+1:sum(Dims[1:i+1]) ]
        St2 = StAcVector2[ sum(Dims[1:i])+1:sum(Dims[1:i+1]) ]
        K = K*exp( -((norm(St1-St2))^2)/2/KernelParam[i+1] )
    end

    return K
end


function Laplace(
        StAcVector1::Vector{Float64},#
        StAcVector2::Vector{Float64},#
        Dims::Vector{Int64},#
        KernelParam::Vector{Float64},#
    )

    St1 = StAcVector1[ 1:Dims[1] ]
    St2 = StAcVector2[ 1:Dims[1] ]
    K = exp( -norm(St1-St2, 1)/KernelParam[1] )

    for i=1:length(Dims)-1
        St1 = StAcVector1[ sum(Dims[1:i])+1:sum(Dims[1:i+1]) ]
        St2 = StAcVector2[ sum(Dims[1:i])+1:sum(Dims[1:i+1]) ]
        K = K*exp( -norm(St1-St2, 1)/KernelParam[i+1] )
    end

    return K
end


function RBF(
        StAcVector1::Vector{Float64},#
        StAcVector2::Vector{Float64},#
        Dims::Vector{Int64},#
        KernelParam::Vector{Float64},#
    )
    
    St1 = StAcVector1[ 1:Dims[1] ]
    St2 = StAcVector2[ 1:Dims[1] ]
    K = exp( -norm(St1-St2)/KernelParam[1] )

    for i=1:length(Dims)-1
        St1 = StAcVector1[ sum(Dims[1:i])+1:sum(Dims[1:i+1]) ]
        St2 = StAcVector2[ sum(Dims[1:i])+1:sum(Dims[1:i+1]) ]
        K = K*exp( -norm(St1-St2)/KernelParam[i+1] )
    end

    return K
end


function PolyRBF(
        StAcVector1::Vector{Float64},#
        StAcVector2::Vector{Float64},#
        Dims::Vector{Int64},#
        KernelParam::Vector{Float64},#
    )

    St1 = StAcVector1[ 1:Dims[1] ]
    St2 = StAcVector2[ 1:Dims[1] ]
    P = ( St1'*St2 )^(KernelParam[1])

    Ac1 = StAcVector1[ sum(Dims[1:2])+1:sum(Dims) ]
    Ac2 = StAcVector2[ sum(Dims[1:2])+1:sum(Dims) ]
    K = exp( -norm(Ac1 - Ac2)/KernelParam[2] )

    return K*P
end
