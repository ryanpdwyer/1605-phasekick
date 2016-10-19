using ToeplitzMatrices
# using PyPlot

function displayln(x)
    display(x)
    println("\n")
end

function toeplitz(kernel, N, M)
    K = length(kernel)
    _eltype = eltype(kernel)
    v1 = zeros(_eltype, N)
    v2 = zeros(_eltype, M)
    v1[1] = kernel[1]
    display(v1)
    v2[1:K] = kernel
    display(v2)
    return Toeplitz(v1, v2)
end

ctrans(x::Toeplitz) = Toeplitz(conj(x.vr), conj(x.vc))

# H = [0.25, 0.5, 0.25]

# Ny = 126
# Nx = 128

# A = toeplitz(H, Ny, Nx)

# x_exact = zeros(Nx)
# inds = Dict(23=>1.2, 45=>0.3, 50=>2.0, 55=>1.0)
# for (i, val) in inds
#     x_exact[i] = val
# end

# y_exact = A * x_exact

# displayln(A)
# displayln(x_exact)
# displayln(y_exact)

# An = zeros(A)
# scales = zeros(x_exact)
# for i = 1:size(A)[2]
#     c = A[:, i]
#     scale = norm(c)
#     scales[i] = scale
#     An[:, i] = c / scale
# end

function omp(A::Toeplitz, y, k)
    inds = Vector{Int}()
    r = copy(y)
    A_ct = ctrans(A)
    for j = 1:k
        global xapprox
        rX = squeeze(abs(A_ct * r''), 2)
        displayln(rX)
        sortedinds = sortperm(rX, rev=true)
        for ind_ in sortedinds
            if !(ind_ in inds)
                push!(inds, ind_)
                break
            end
        end
        Aapprox = A[:, inds]
        # Somehow I was writing the wrong number on top!
        xapprox = Aapprox \ y 
        r[:] = y - Aapprox * xapprox
    end
    x = zeros(size(A)[2])
    x[inds] = xapprox
    return x
end

# xr = omp(A, y_exact, 6)

# plot(x_exact)
# plot(xr)


