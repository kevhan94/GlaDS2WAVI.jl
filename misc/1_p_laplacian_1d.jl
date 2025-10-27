# Step 1: accelerated PT
using Printf, CairoMakie

function main()
    # physics
    lx  = 1.0
    p   = 3 / 2
    ∇ϕr = 1e-4
    m   = 1.0
    # numerics
    nx      = 100
    dx      = lx / nx
    maxiter = 2000nx
    ncheck  = 10nx
    etol    = 1e-8
    # preprocessing
    xc = LinRange(dx / 2, lx - dx / 2, nx)
    # arrays
    ϕ  = zeros(nx + 2)
    ∇ϕ = zeros(nx + 1)
    d  = zeros(nx + 1)
    q  = zeros(nx + 1)
    r  = zeros(nx)
    z  = zeros(nx)
    s  = zeros(nx)
    # plotting
    fig = Figure()
    axs = Axis(fig[1, 1])
    lns = lines!(axs, xc, ϕ[2:end-1])
    # iterative loop
    for iter in 1:maxiter
        # compute diffusivity
        @. ∇ϕ = (ϕ[2:end] - ϕ[1:end-1]) / dx
        @. d  = abs(∇ϕ + ∇ϕr)^(p - 2)
        # compute flux
        @. q = -d * ∇ϕ
        # compute residual
        @. r = -(q[2:end] - q[1:end-1]) / dx + m
        # compute preconditioned residual
        @. z = dx^2 * r / (d[1:end-1] + d[2:end])
        # compute damping factor
        β = 1 - 0.5 / nx
        # compute search direction
        @. s = s * β + z
        # compute step size
        α = 1.0
        # update solution
        @. ϕ[2:end-1] += α * s
        # boundary conditions
        ϕ[1]   = 0
        ϕ[end] = ϕ[end-1]
        # check convergence
        if iter % ncheck == 0
            err = maximum(abs.(r))
            @printf("   iter = %.1f × nx, err = %1.3e\n", iter / nx, err)
            if err < etol
                break
            end
        end
    end
    lns[2] = ϕ[2:end-1]
    display(fig)
    return
end

main()
