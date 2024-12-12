using TSVD
using LinearAlgebra
using MIRT

function ALPCAH(Y::Matrix, k::Int, λr::Real; μ::Real=0.01, ρ::Real=1.0, alpcahIter::Int=1000, varfloor::Real=1e-9)
    U_init = svd(Y).U[:,1:k]
    X = deepcopy(U_init*U_init'*Y)
    Z = deepcopy(Y-X)
    var = grouplessVarianceUpdate(Y, X; varfloor=varfloor)
    Π = Diagonal(var.^-1)
    Λ = sign.(Y)
    Λ = deepcopy(Λ ./ (max(opnorm(Λ), (1/λr)*norm(Λ, Inf))))
    #alpha = LinRange(0.0, λr, alpcahIter)
    X = zeros(size(Y))
    Z = zeros(size(Y))
    var = ones(length(var))
    for i = 1:alpcahIter
        X = TSVT(Y-Z+(1/μ)*Λ, λr/μ,k) # TSVT(Y-Z+(1/μ)*Λ, λr/μ,k)
        Z = μ*(Y-X+(1/μ)*Λ)*inv(Π+μ*I)
        Λ = Λ + μ*(Y-X-Z)
        var = grouplessVarianceUpdate(Y, X; varfloor=varfloor)
        Π = Diagonal(var.^-1)
        μ = ρ*μ
    end
    U = svd(X).U[:,1:k]   
    return U
end

function ALPCAH_KNOWN(Y::Matrix, k::Int, λr::Real, var::Vector; μ::Real=0.01, ρ::Real=1.0, alpcahIter::Int=1000, varfloor::Real=1e-9)
    U_init = svd(Y).U[:,1:k]
    X = deepcopy(U_init*U_init'*Y)
    Z = deepcopy(Y-X)
    Π = Diagonal(var.^-1)
    Λ = sign.(Y)
    Λ = deepcopy(Λ ./ (max(opnorm(Λ), (1/λr)*norm(Λ, Inf))))
    X = zeros(size(Y))
    Z = zeros(size(Y))
    #var = ones(length(var))
    for i = 1:alpcahIter
        X = TSVT(Y-Z+(1/μ)*Λ, λr/μ,k) # TSVT(Y-Z+(1/μ)*Λ, λr/μ,k)
        Z = μ*(Y-X+(1/μ)*Λ)*inv(Π+μ*I)
        Λ = Λ + μ*(Y-X-Z)
        μ = ρ*μ
    end
    U = svd(X).U[:,1:k]   
    return U
end

function LR_ALPCAH(Y::Matrix,d::Int; varfloor::Real=1e-9, alpcahIter::Int = 1000, fastCompute::Bool=false)
    """
    Returns subspace basis given data matrix Y and specified dimension of basis
    by using a matrix factorized version of ALPCAH that treats each point as
    having its own noise variance for heteroscedastic data

    Input:
    Y is DxN data matrix of N data points and ambient dimension D
    d is integer of subspace (must be known or predicted before hand)
    varfloor is opt. parameter to keep noise variances from pushing to 0
    alpcahIter is opt. integer specifing how many iterations to run the algorithm

    Optional:
    fastCompute (bool) determines whether to use partial svd method (Krylov) or
    regular compact SVD. Multithreading safe using TSVD instead of Arpack.

    Output:
    U is Dxd subspace basis of d orthonormal vectors given ambient dimension D
    """
    rank = d
    D,N = size(Y)

    T = svd(Y)
    L = T.U[:,1:rank]*Diagonal(sqrt.(T.S[1:rank]))
    R = T.V[:,1:rank]*Diagonal(sqrt.(T.S[1:rank]))

    # variance method initialization
    v = grouplessVarianceUpdate(Y, L*R'; varfloor=varfloor)
    Π = Diagonal(v.^-1)

    for i=1:alpcahIter
        # left right updates
        L = Y*Π*R*inv(R'*Π*R)
        R = Y'*L*inv(L'*L)
        # variance updates
        v = grouplessVarianceUpdate(Y, L*R'; varfloor=varfloor)
        Π = Diagonal(v.^-1)
    end
    # extract left vectors from L
    U = svd(L).U
    #U = tsvd(L,rank)[1]
    return U
end

function LR_ALPCAH_KNOWN(Y::Matrix,d::Int, w::Vector; varfloor::Real=1e-9, alpcahIter::Int = 100, fastCompute::Bool=false)
    D,N = size(Y)

    T = svd(Y)
    L = T.U[:,1:d]*Diagonal(sqrt.(T.S[1:d]))
    R = T.V[:,1:d]*Diagonal(sqrt.(T.S[1:d]))
    Π = Diagonal(w.^-1)

    for i=1:alpcahIter
        # left right updates
        L = Y*Π*R*inv(R'*Π*R)
        R = Y'*L*inv(L'*L)
    end
    # extract left vectors from L
    U = svd(L).U
    return U
end

function grouplessVarianceUpdate(Y::Matrix, X::Matrix; varfloor::Real=1e-9)
    D= size(Y)[1]
    Π = (1/D)*norm.(eachcol(Y - X)).^2
    return max.(Π, varfloor)
end

# Only works for two groups
function groupedVarianceUpdate(Y::Matrix, X::Matrix, goodpts::Int; varfloor::Real=1e-9)
    D,N = size(Y)
    Π = zeros(N)
    Π[1:goodpts] .= (1/(D*goodpts))*norm(Y[:,1:goodpts]-X[:,1:goodpts],2)^2
    Π[(goodpts+1):end] .= (1/(D*(N-goodpts)))*norm(Y[:,(goodpts+1):end]-X[:,(goodpts+1):end],2)^2
    return max.(Π, varfloor)
end

function varianceGroupResult(Y::Matrix, X::Matrix, goodpts::Int; varfloor::Real=1e-9)
    D,N = size(Y)
    Π = zeros(2)
    Π[1] = (1/(D*goodpts))*norm(Y[:,1:goodpts]-X[:,1:goodpts],2)^2
    Π[2] = (1/(D*(N-goodpts)))*norm(Y[:,(goodpts+1):end]-X[:,(goodpts+1):end],2)^2
    return max.(Π, varfloor)
end

function TSVT(A::Matrix,λ::Real,k::Int)
    U,S,V = svd(A)
    S[(k+1):end] = softTresh.(S[(k+1):end],λ)
    return U*Diagonal(S)*V'
end

function SVST(X::Matrix,t::Real)
    U,S,V = svd(X)
    return U*Diagonal(softTresh.(S, t))*V'
end

function softTresh(x::Real, λ::Real)
    return sign(x) * max(abs(x) - λ, 0)
end

function PCA(A::Matrix, d::Int)
    return svd(A).U[:,1:d]
end

function ALPCAH_GROUPED(Y::Matrix, k::Int, λr::Real; μ::Real=0.01, ρ::Real=1.0, alpcahIter::Int=1000, varfloor::Real=1e-9, goodpts::Int=25)
    U_init = svd(Y).U[:,1:k]
    X = deepcopy(U_init*U_init'*Y)
    Z = deepcopy(Y-X)
    var = groupedVarianceUpdate(Y, X, goodpts; varfloor=varfloor)
    Π = Diagonal(var.^-1)
    Λ = sign.(Y)
    Λ = deepcopy(Λ ./ (max(opnorm(Λ), (1/λr)*norm(Λ, Inf))))
    #alpha = LinRange(0.0, λr, alpcahIter)
    X = zeros(size(Y))
    Z = zeros(size(Y))
    var = ones(length(var))
    for i = 1:alpcahIter
        X = TSVT(Y-Z+(1/μ)*Λ, λr/μ,k) # TSVT(Y-Z+(1/μ)*Λ, λr/μ,k)
        Z = μ*(Y-X+(1/μ)*Λ)*inv(Π+μ*I)
        Λ = Λ + μ*(Y-X-Z)
        var = groupedVarianceUpdate(Y, X, goodpts; varfloor=varfloor)
        Π = Diagonal(var.^-1)
        μ = ρ*μ
    end
    U = svd(X).U[:,1:k]   
    return U
end

function LR_ALPCAH_GROUPED(Y::Matrix,d::Int; varfloor::Real=1e-9, alpcahIter::Int = 1000, fastCompute::Bool=false, goodpts::Int=25)
    D,N = size(Y)
    rank = d
    
    T = svd(Y)
    L = T.U[:,1:rank]*Diagonal(sqrt.(T.S[1:rank]))
    R = T.V[:,1:rank]*Diagonal(sqrt.(T.S[1:rank]))

    # variance method initialization
    v = groupedVarianceUpdate(Y, L*R', goodpts; varfloor=varfloor)
    Π = Diagonal(v.^-1)

    for i=1:alpcahIter
        # left right updates
        L = Y*Π*R*inv(R'*Π*R)
        R = Y'*L*inv(L'*L)
        # variance updates
        v = groupedVarianceUpdate(Y, L*R', goodpts; varfloor=varfloor)
        Π = Diagonal(v.^-1)
    end
    # extract left vectors from L
    U = svd(L).U
    #U = tsvd(L,rank)[1]
    return U
end

function ALPCAH_NUCLEAR(Y::Matrix, k::Int, λr::Real; μ::Real=0.01, ρ::Real=1.0, alpcahIter::Int=1000, varfloor::Real=1e-9)
    U_init = svd(Y).U[:,1:k]
    X = deepcopy(U_init*U_init'*Y)
    Z = deepcopy(Y-X)
    var = grouplessVarianceUpdate(Y, X; varfloor=varfloor)
    Π = Diagonal(var.^-1)
    Λ = sign.(Y)
    Λ = deepcopy(Λ ./ (max(opnorm(Λ), (1/λr)*norm(Λ, Inf))))
    #alpha = LinRange(0.0, λr, alpcahIter)
    X = zeros(size(Y))
    Z = zeros(size(Y))
    var = ones(length(var))
    for i = 1:alpcahIter
        X = SVST(Y-Z+(1/μ)*Λ, λr/μ) 
        Z = μ*(Y-X+(1/μ)*Λ)*inv(Π+μ*I)
        Λ = Λ + μ*(Y-X-Z)
        var = grouplessVarianceUpdate(Y, X; varfloor=varfloor)
        Π = Diagonal(var.^-1)
        μ = ρ*μ
    end
    U = svd(X).U[:,1:k]   
    return U
end

function ALPCAH_NUCLEAR_KNOWN(Y::Matrix, k::Int, λr::Real, var::Vector; μ::Real=0.01, ρ::Real=1.0, alpcahIter::Int=1000, varfloor::Real=1e-9)
    U_init = svd(Y).U[:,1:k]
    X = deepcopy(U_init*U_init'*Y)
    Z = deepcopy(Y-X)
    Π = Diagonal(var.^-1)
    Λ = sign.(Y)
    Λ = deepcopy(Λ ./ (max(opnorm(Λ), (1/λr)*norm(Λ, Inf))))
    #X = zeros(size(Y))
    #Z = zeros(size(Y))
    #var = ones(length(var))
    for i = 1:alpcahIter
        X = SVST(Y-Z+(1/μ)*Λ, λr/μ) 
        Z = μ*(Y-X+(1/μ)*Λ)*inv(Π+μ*I)
        Λ = Λ + μ*(Y-X-Z)
        μ = ρ*μ
    end
    U = svd(X).U[:,1:k]   
    return U
end

function estimateRank(A::Matrix; rankMethod::Symbol=:flippa, quantileAmount::Real=0.95, flipTrials::Int=10)
    """
    Estimates rank of a matrix by using methods listed

    Input:
    A is DxN matrix where D is ambient dimension and N is number of points

    Optional:
    rankMethod (symbol) can be :flippa or :eigengap. FlipPA method works
        best especially when the data is heteroscedastic. Much harder to differentiate
        signal components from noise in this setting.
    quantileAmount (real) between 0.0 and 1.0 is a quantile metric for the
        trials done in flippa. Best to leave 0.95 or 1.0 for confidence reasons.
    flipTrials (integer) describes how many trials of permutations to do
        for flippa method. Higher is better, do as many as computation time allows.

    Output:
    Integer estimate of the subspace associated with low rank matrix A
    """
    if rankMethod === :flippa
        return flippa(A; quantile=quantileAmount, trials=flipTrials)
    elseif rankMethod === :eigengap
        return argmax( -1*diff( reverse( eigen(A*A').values ) ) )
    end
end