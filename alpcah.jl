using TSVD
using LinearAlgebra
using MIRT

# ALPCAH MAIN WRAPPER FUNCTION

function ALPCAH(Y::Matrix, rank::Int; methodType::Symbol = :ALPCAH_ALTMIN, alpcahIter::Int = 1000, apgdIter::Int = 5, vknown::Bool = false, varfloor::Real=1e-9, λr::Real=1e6, varianceMethod::Symbol=:groupless, v::Vector=ones(3), μ::Real=0.01, ρ::Real=1.0)
    U = 0
    if methodType === :ALPCAH_ALTMIN
        if vknown == false
            U = ALPCAH_UNKNOWN_ALTMIN(Y, rank; varfloor=varfloor, alpcahIter=alpcahIter, varianceMethod=varianceMethod)
        else
            U = ALPCAH_KNOWN_ALTMIN(Y, rank, v; alpcahIter=alpcahIter)
        end
    end
    if methodType === :ALPCAH_ADMM
        if vknown == false
            U = ALPCAH_UNKNOWN_ADMM(Y, rank, λr; μ=μ, ρ=ρ, alpcahIter=alpcahIter, varfloor=varfloor)
        else
            U = ALPCAH_KNOWN_ADMM(Y, rank, v; μ=μ, ρ=ρ, λr=λr, alpcahIter=alpcahIter)
        end
    end
    if methodType === :ALPCAH_APGD
        if vknown == false
            U = ALPCAH_UNKNOWN_APGD(Y, rank, λr; alpcahIter=alpcahIter, apgdIter=apgdIter, varianceMethod=varianceMethod, varfloor=varfloor)
        else
            U = ALPCAH_KNOWN_APGD(Y, rank, v; alpcahIter=alpcahIter, λr=λr)
        end
    end
    X = U*U'*Y
    v = grouplessVarianceUpdate(Y, X; varfloor=varfloor)
    return U, X, v
end

# FAST, LOW-MEMORY IMPLEMENTATIONS OF ALPCAH X = LR' MATRIX FACTORIZATION USING ALTMIN

function ALPCAH_UNKNOWN_ALTMIN(Y::Matrix, rank::Int; varfloor::Real=1e-9, alpcahIter::Int = 5, varianceMethod::Symbol = :groupless)
    D,N = size(Y)
    v = zeros(N)
    # Krylov-based Lanczos Algorithm
    T = tsvd(Y, rank)
    L = T[1]*Diagonal(sqrt.(T[2]))
    R = T[3]*Diagonal(sqrt.(T[2]))
    # variance method initialization
    if varianceMethod === :groupless
        v = grouplessVarianceUpdate(Y, L*R'; varfloor=varfloor)
    else
        #v = groupedVarianceUpdate(Y, L*R' ; varfloor=varfloor)
    end
    Π = Diagonal(v.^-1)
    for i=1:alpcahIter
        # left right updates
        L = Y*Π*R*inv(R'*Π*R)
        R = Y'*L*inv(L'*L)
        # variance updates
        if varianceMethod === :groupless
            v = grouplessVarianceUpdate(Y, L*R'; varfloor=varfloor)
        else
            #v = groupedVarianceUpdate(Y, L*R' ; varfloor=varfloor)
        end
        Π = Diagonal(v.^-1)
    end
    # extract left vectors from L
    U = svd(L).U
    return U
end

function ALPCAH_KNOWN_ALTMIN(Y::Matrix, rank::Int, v::Vector; alpcahIter::Int = 10)
    # Krylov-based Lanczos Algorithm
    T = tsvd(Y, rank)
    L = T[1]*Diagonal(sqrt.(T[2]))
    R = T[3]*Diagonal(sqrt.(T[2]))
    # variance inverse matrix
    Π = Diagonal(v.^-1)
    for i=1:alpcahIter
        # left right updates
        L = Y*Π*R*inv(R'*Π*R)
        R = Y'*L*inv(L'*L)
    end
    # extract left vectors from L
    U = svd(L).U
    return U
end

# GROUPLESS VARIANCE UPDATE

function grouplessVarianceUpdate(Y::Matrix, X::Matrix; varfloor::Real=1e-9)
    D= size(Y)[1]
    Π = (1/D)*norm.(eachcol(Y - X)).^2
    return max.(Π, varfloor)
end

# PROXIMAL OPERATORS

function TSVT(A::Matrix,λ::Real,k::Int)
    U,S,V = svd(A)
    S[(k+1):end] = softTresh.(S[(k+1):end],λ)
    return U*Diagonal(S)*V'
end

function softTresh(x::Real, λ::Real)
    return sign(x) * max(abs(x) - λ, 0)
end

# APGD IMPLEMENTATIONS OF ALPCAH

function ALPCAH_KNOWN_APGD(Y::Matrix, rank::Int, v::Vector; alpcahIter::Int = 1000, λr::Real = 1e6)
    Π = v.^-1
    Lf = maximum(Π)
    Π = Diagonal(Π)
    U = tsvd(Y, rank)[1]
    X = U*U'*Y
    grad = X -> -1*(Y-X)*Π
    g_prox = (z,c) -> TSVT(z, c*λr, rank)
    X, _ = pogm_restart(X, x -> 0, grad, Lf ; mom=:fpgm, g_prox, niter=alpcahIter);
    U = tsvd(X, rank)[1]
    return U
end

function ALPCAH_UNKNOWN_APGD(Y::Matrix, rank::Int, λr::Real; alpcahIter::Int = 200, apgdIter::Int = 5, varianceMethod::Symbol = :groupless, varfloor::Real=1e-9)
    D, N = size(Y)
    v = zeros(N)
    # Krylov-based Lanczos Algorithm
    U = tsvd(Y, rank)[1]
    X = U*U'*Y
    #variance method initialization
    if varianceMethod === :groupless
        v = grouplessVarianceUpdate(Y, X; varfloor=varfloor)
    else
        #v = groupedVarianceUpdate(Y, L*R' ; varfloor=varfloor)
    end
    Π = v.^-1
    Lf = maximum(Π)
    Π = Diagonal(Π)
    X = zeros(size(Y)) # weirdly necessary to get lower error
    grad = K -> -1*(Y-K)*Π
    #alpha = LinRange(0.0, λr, alpcahIter)
    for i=1:alpcahIter
        # apgd
        g_prox = (z,c) -> TSVT(z, c*λr, rank) # TSVT(z, c*alpha[i], rank)
        X, _ = pogm_restart(X, x -> 0, grad, Lf ; mom=:fpgm, g_prox, niter=apgdIter);
        # variance updates
        if varianceMethod === :groupless
            v = grouplessVarianceUpdate(Y, X; varfloor=varfloor)
        else
            #v = groupedVarianceUpdate(Y, L*R' ; varfloor=varfloor)
        end
        Π = v.^-1
        Lf = maximum(Π)
        Π = Diagonal(Π)
    end
    # extract left vectors from X
    U = tsvd(X, rank)[1]
    return U
end

# ADMM IMPLEMENTATIONS OF ALPCAH

function ALPCAH_UNKNOWN_ADMM(Y::Matrix, k::Int, λr::Real; μ::Real=0.01, ρ::Real=1.0, alpcahIter::Int=1000, varfloor::Real=1e-9)
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

function ALPCAH_KNOWN_ADMM(Y::Matrix, k::Int, v::Vector; μ::Real=0.01, ρ::Real=1.0, λr::Real=1e6 ,alpcahIter::Int = 1000)
    X = zeros(size(Y))
    Z = zeros(size(Y))
    Π = Diagonal(v.^-1)
    Λ = sign.(Y)
    Λ = deepcopy(Λ ./ (max(opnorm(Λ), (1/λr)*norm(Λ, Inf))))
    for i = 1:alpcahIter
        X = TSVT(Y-Z+(1/μ)*Λ, λr/μ,k)
        Z = μ*(Y-X+(1/μ)*Λ)*inv(Π+μ*I)
        Λ = Λ + μ*(Y-X-Z)
        μ = ρ*μ
    end
    U = svd(X).U[:,1:k]   
    return U
end

function QR_ALPCAH(Y, niter, d)
    Q,R = qr(Y)
    Q = Q[:,1:d]
    R = R[1:d,:]
    v = grouplessVarianceUpdate(Y, Q*R)
    for i=1:niter
        U,_,V = svd(Y*Diagonal(v.^-1)*R')
        Q = U*V'
        R = Q'*Y
        v = grouplessVarianceUpdate(Y, Q*R)
    end
    return Q
end

function PCA(A::Matrix, d::Int)
    return svd(A).U[:,1:d]
end