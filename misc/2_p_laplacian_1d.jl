# Step 2: non-linear conjugate gradient
using Printf, CairoMakie, Enzyme, LinearAlgebra

@views function residual!(r, ∇ϕ, d, q, ϕ, ∇ϕr, m, p, dx)
    # boundary conditions
    ϕ[1]   = 0
    ϕ[end] = ϕ[end-1]
    # compute diffusivity
    @. ∇ϕ = (ϕ[2:end] - ϕ[1:end-1]) / dx
    @. d  = abs(∇ϕ + ∇ϕr)^(p - 2)
    # compute flux
    @. q = -d * ∇ϕ
    # compute residual
    @. r = -(q[2:end] - q[1:end-1]) / dx + m
    return
end

@views function preconditioner!(z, r, d, dx)
    @. z = dx^2 * r / (d[1:end-1] + d[2:end])
    return
end

@views function main()
    # physics
    lx  = 1.0
    p   = 3 / 2
    ∇ϕr = 1e-4
    m   = 1.0
    # numerics
    nx      = 100
    dx      = lx / nx
    maxiter = 1000nx
    ncheck  = 10nx
    etol    = 1e-8
    # preprocessing
    xc = LinRange(dx / 2, lx - dx / 2, nx)
    # arrays
    ϕ  = zeros(nx + 2)
    ϕ0 = zeros(nx + 2)
    ∇ϕ = zeros(nx + 1)
    d  = zeros(nx + 1)
    q  = zeros(nx + 1)
    r  = zeros(nx)
    z  = zeros(nx)
    s  = zeros(nx)
    # shadows
    ϕ̄  = make_zero(ϕ)
    ∇ϕ̄ = make_zero(∇ϕ)
    d̄  = make_zero(d)
    q̄  = make_zero(q)
    r̄  = make_zero(r)
    # plotting
    fig = Figure()
    axs = Axis(fig[1, 1])
    lns = lines!(axs, xc, ϕ[2:end-1])
    # iterative loop
    residual!(r, ∇ϕ, d, q, ϕ, ∇ϕr, m, p, dx)
    # compute preconditioned residual
    preconditioner!(z, r, d, dx)
    # compute search direction
    @. s = z
    rz0  = dot(r, z)
    z0   = copy(z)
    for iter in 1:maxiter
        @. ϕ0 = ϕ
        # line search
        for ils in 1:2
            # JVP
            make_zero!(r̄)
            make_zero!(∇ϕ̄)
            make_zero!(d̄)
            make_zero!(q̄)
            make_zero!(ϕ̄)
            @. ϕ̄[2:end-1] = s
            Enzyme.autodiff(set_runtime_activity(Enzyme.Forward),
                            residual!,
                            Duplicated(r, r̄),
                            Duplicated(∇ϕ, ∇ϕ̄),
                            Duplicated(d, d̄),
                            Duplicated(q, q̄),
                            Duplicated(ϕ, ϕ̄),
                            Const(∇ϕr),
                            Const(m),
                            Const(p),
                            Const(dx))
            # compute step size
            α = -0.8rz0 / dot(s, r̄)
            # update solution
            @. ϕ[2:end-1] = ϕ0[2:end-1] + α * s
        end
        # compute residual
        residual!(r, ∇ϕ, d, q, ϕ, ∇ϕr, m, p, dx)
        # check convergence
        if iter % ncheck == 0
            err = maximum(abs.(r))
            @printf("   iter = %.1f × nx, err = %1.3e\n", iter / nx, err)
            if err < etol
                break
            end
        end
        @. z0 = z
        # compute preconditioned residual
        preconditioner!(z, r, d, dx)
        # compute damping factor
        rz  = dot(r, z)
        β   = max((rz - dot(r, z0)) / rz0, 0)
        rz0 = rz
        # compute search direction
        @. s = s * β + z
    end
    lns[2] = ϕ[2:end-1]
    display(fig)
    return
end

main()
