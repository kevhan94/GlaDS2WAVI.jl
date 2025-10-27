# Step 4: Newton's method conjugate gradient with diagonal preconditioner
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

@views function preconditioner!(P, e, r̄, r, ∇ϕ, ∇ϕ̄, d, d̄, q, q̄, ϕ, ϕ̄, ∇ϕr, m, p, dx)
    for ic in 1:2
        @. e = 0
        @. e[ic:2:end] = 1
        jvp!(e, r̄, r, ∇ϕ, ∇ϕ̄, d, d̄, q, q̄, ϕ, ϕ̄, ∇ϕr, m, p, dx)
        @. P[ic:2:end] = r̄[ic:2:end]
    end
    @. P = inv(abs(P))
    return
end

@views function jvp!(v, r̄, r, ∇ϕ, ∇ϕ̄, d, d̄, q, q̄, ϕ, ϕ̄, ∇ϕr, m, p, dx)
    make_zero!(∇ϕ̄)
    make_zero!(d̄)
    make_zero!(q̄)
    make_zero!(ϕ̄)
    @. ϕ̄[2:end-1] = v
    Enzyme.autodiff(Enzyme.Forward,
                    residual!,
                    DuplicatedNoNeed(r, r̄),
                    DuplicatedNoNeed(∇ϕ, ∇ϕ̄),
                    DuplicatedNoNeed(d, d̄),
                    DuplicatedNoNeed(q, q̄),
                    DuplicatedNoNeed(ϕ, ϕ̄),
                    Const(∇ϕr),
                    Const(m),
                    Const(p),
                    Const(dx))
    return
end

@views function increment!(δϕ, r, r̄, k, z, s, ∇ϕ, ∇ϕ̄, d, d̄, q, q̄, ϕ, ϕ̄, P, e, ∇ϕr, m, p, maxiter, ncheck, cgtol, nx, dx)
    @. δϕ = 0
    # compute linearised residual
    jvp!(δϕ, r̄, r, ∇ϕ, ∇ϕ̄, d, d̄, q, q̄, ϕ, ϕ̄, ∇ϕr, m, p, dx)
    @. k = r̄ + r
    # precompute diagonal preconditioner
    preconditioner!(P, e, r̄, r, ∇ϕ, ∇ϕ̄, d, d̄, q, q̄, ϕ, ϕ̄, ∇ϕr, m, p, dx)
    # compute preconditioned linearised residual
    @. z = P * k
    # compute search direction
    @. s = z
    kz0  = dot(k, z)
    for iter in 1:maxiter
        # JVP
        jvp!(s, r̄, r, ∇ϕ, ∇ϕ̄, d, d̄, q, q̄, ϕ, ϕ̄, ∇ϕr, m, p, dx)
        # compute step size
        α = -kz0 / dot(s, r̄)
        # update solution
        @. δϕ += α * s
        # compute linearized residual
        @. k += α * r̄
        # check convergence
        if iter % ncheck == 0
            jvp!(δϕ, r̄, r, ∇ϕ, ∇ϕ̄, d, d̄, q, q̄, ϕ, ϕ̄, ∇ϕr, m, p, dx)
            @. k = r̄ + r
            err = (norm(k) / norm(r), α * norm(s) / norm(δϕ))
            @printf("   iter = %.2f × nx, err = [abs = %1.3e, rel = %1.3e]\n", iter / nx, err...)
            if any(err .< cgtol)
                break
            end
        end
        # compute preconditioned linearised residual
        @. z = P * k
        # compute damping factor
        kz  = dot(k, z)
        β   = kz / kz0
        kz0 = kz
        # compute search direction
        @. s = s * β + z
    end
    return
end

@views function main()
    # physics
    lx  = 1.0
    p   = 3 / 2
    ∇ϕr = 1e-4
    m   = 1.0
    # numerics
    nx      = 1000
    dx      = lx / nx
    maxiter = 10nx
    ncheck  = ceil(Int, 0.25nx)
    cgtol   = 1e-10
    nltol   = 1e-6
    maxnewt = 50
    # preprocessing
    xc = LinRange(dx / 2, lx - dx / 2, nx)
    # arrays
    ϕ  = zeros(nx + 2)
    ∇ϕ = zeros(nx + 1)
    d  = zeros(nx + 1)
    q  = zeros(nx + 1)
    δϕ = zeros(nx)
    r  = zeros(nx)
    z  = zeros(nx)
    s  = zeros(nx)
    k  = zeros(nx)
    P  = zeros(nx)
    e  = zeros(nx)
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
    for inewt in 1:maxnewt
        # compute nonlinear residual
        residual!(r, ∇ϕ, d, q, ϕ, ∇ϕr, m, p, dx)
        # check convergence
        err = norm(r) / length(r)
        @printf("inewt = %d, err = %1.3e\n", inewt, err)
        if err < nltol
            break
        end
        increment!(δϕ, r, r̄, k, z, s, ∇ϕ, ∇ϕ̄, d, d̄, q, q̄, ϕ, ϕ̄, P, e, ∇ϕr, m, p, maxiter, ncheck, cgtol, nx, dx)
        # update solution
        γ = 0.9
        @. ϕ[2:end-1] += γ * δϕ
    end
    lns[2] = ϕ[2:end-1]
    display(fig)
    return
end

main()
