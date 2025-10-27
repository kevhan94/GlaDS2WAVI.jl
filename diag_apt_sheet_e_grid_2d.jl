"""
SHIMP Suite A (2D on Arakawa E-grid)
    de Fleurian et al. (2018): "SHMIP The subglacial hydrology model intercomparison Project"

Physics:
- Conservation equation:
    ∂ₜh + ∇·q = m, where q = -k h^α |∇ϕ|^(β-2) ∇ϕ

- Time evolution of water sheet thickness:
    ∂ₜh = w - v

- No englacial storage is included (englacial void fraction e_v = 0).

- Boundary conditions:
    Dirichlet: ϕ = 0 at the left boundary
    Neumann: ∂ₙϕ = 0 at the right, upper and lower boundary

Key features:
    - Accelerated PT solver with Auto-tuning
    - Diagonal preconditioner via efficient extraction of the diag(Jϕ) with graph coloring
        => naive inverse Laplacian as preconditioner diverges
"""

using Printf
using CairoMakie
using Enzyme, LinearAlgebra

using TimerOutputs
const to = TimerOutput()

@views av(x) = @. 0.5 * (x[1:end-1] + x[2:end])

@views function residual!(r, ϕ, ∂ₓϕ, ∂ᵧϕ, qx, qy, D, ρigH, h, k, α, β, u_b, h_r, l_r, Ã, glens_n, dx, dy, m)
    # p-Laplacian
    # q = -kh^α|∇ϕ|^(β-2)∇ϕ
    # where D := kh^α|∇ϕ|^(β-2)
    # boundary conditions
    @. ϕ.c[1, :]   = 0
    @. ϕ.c[end, :] = ϕ.c[end-1, :]
    @. ϕ.c[:, 1]   = ϕ.c[:, 2]
    @. ϕ.c[:, end] = ϕ.c[:, end-1]

    # compute gradients (Vx, Vy nodes)
    # part c
    @. ∂ₓϕ.c = (ϕ.c[2:end, :] - ϕ.c[1:end-1, :]) / dx
    @. ∂ᵧϕ.c = (ϕ.c[:, 2:end] - ϕ.c[:, 1:end-1]) / dy

    # part v
    @. ∂ₓϕ.v[2:end-1, :] = (ϕ.v[2:end, :] - ϕ.v[1:end-1, :]) / dx
    @. ∂ᵧϕ.v[:, 2:end-1] = (ϕ.v[:, 2:end] - ϕ.v[:, 1:end-1]) / dy

    # compute diffusivity
    ∇ϕr = 1e-12
    @. D.vx = (k * (h.c[1:end-1, :])^α) * abs(∂ₓϕ.c^2 + ∂ᵧϕ.v^2 + ∇ϕr)^((β - 2) / 2)
    @. D.vy[2:end, :] = k * (h.v)^α
    @. D.vy *= abs(∂ₓϕ.v^2 + ∂ᵧϕ.c^2 + ∇ϕr)^((β - 2) / 2)

    # update upper and lower part of each cell
    @. qx.c[2:end-1, :] = -D.vx * ∂ₓϕ.c
    @. qy.v = -D.vx * ∂ᵧϕ.v

    # update left and right part of each cell
    @. qy.c[:, 2:end-1] = -D.vy * ∂ᵧϕ.c
    @. qx.v = -D.vy * ∂ₓϕ.v

    # compute residual
    # r = m - ∇·q + v - w
    @. r.c = -((qx.c[2:end, :] - qx.c[1:end-1, :]) / dx + (qy.c[:, 2:end] - qy.c[:, 1:end-1]) / dy) + m +
             (Ã * h.c * abs(ρigH.c - ϕ.c)^(glens_n - 1) * abs(ρigH.c - ϕ.c)) -
             (max(u_b / l_r * (h_r - h.c), 0.0))
    @. r.v = -((qx.v[2:end, :] - qx.v[1:end-1, :]) / dx + (qy.v[:, 2:end] - qy.v[:, 1:end-1]) / dy) + m +
             (Ã * h.v * abs(ρigH.v - ϕ.v)^(glens_n - 1) * abs(ρigH.v - ϕ.v)) -
             (max(u_b / l_r * (h_r - h.v), 0.0))

    return
end

@views function jvp!(v, r̄, r, ϕ, ϕ̄, ∂ₓϕ, ∂ₓϕ̄, ∂ᵧϕ, ∂ᵧϕ̄, qx, qx̄, qy, qȳ, D, D̄, ρigH, ρigH̄, h, h̄, k, α, β, u_b, h_r, l_r, Ã, glens_n, dx, dy, m)
    # Set the input vector for cell and vertex components
    @. ϕ̄.c = v.c
    @. ϕ̄.v = v.v

    # Use Enzyme's forward mode AD to compute the JVP
    Enzyme.autodiff(Enzyme.Forward,
                    residual!,
                    DuplicatedNoNeed(r, r̄),
                    DuplicatedNoNeed(ϕ, ϕ̄),
                    DuplicatedNoNeed(∂ₓϕ, ∂ₓϕ̄),
                    DuplicatedNoNeed(∂ᵧϕ, ∂ᵧϕ̄),
                    DuplicatedNoNeed(qx, qx̄),
                    DuplicatedNoNeed(qy, qȳ),
                    DuplicatedNoNeed(D, D̄),
                    DuplicatedNoNeed(ρigH, ρigH̄),
                    DuplicatedNoNeed(h, h̄),
                    Const(k),
                    Const(α),
                    Const(β),
                    Const(u_b),
                    Const(h_r),
                    Const(l_r),
                    Const(Ã),
                    Const(glens_n),
                    Const(dx),
                    Const(dy),
                    Const(m))
    return nothing
end

"""
Extract diag(Jϕ) from J_CC, J_VV blocks.
- The coloring is based on diagonal blocks J_CC, J_VV of the Jacobian

Returns two tuples of indices, each contains two colors:
- Ic: Two groups of center grid indices for diag(J_cc)
- Iv: Two groups of vertex grid indices for diag(J_vv)
"""
function coloring(nx, ny)
    Nx_selected = LinearIndices((nx, ny))[2:end-1, 2:end-1]
    Ny_selected = LinearIndices((nx - 1, ny - 1))[2:end-1, 2:end-1]

    # indices for coloring of center grid
    mC = iseven.((2:nx-1) .+ (2:ny-1)')
    Ic = (vec(Nx_selected[mC]), vec(Nx_selected[.!mC]))

    # indices for coloring of vertex grid
    mV = iseven.((2:nx-2) .+ (2:ny-2)')
    Iv = (vec(Ny_selected[mV]), vec(Ny_selected[.!mV]))

    return Ic, Iv
end

function sheet_2d_e_grid()
    # numerics
    nx, ny = 128, 4
    lx, ly = 100e3, 20e3 # 100 x 20 [km]

    # derived parameters
    @show dx, dy = lx / nx, ly / ny

    # APT iterative loop
    ncheck      = 10nx
    maxiter     = 10nx^2
    etol        = 1e-8
    do_autotune = true
    n_autotune  = 10

    # physics
    shmip = [7.93e-11, 1.59e-9, 5.79e-9, 2.5e-8, 4.5e-8, 5.79e-7]
    m     = shmip[1]
    k     = 0.005    # Sheet conductivity
    α     = 5 / 4    # 1st turbulent flow exponent (GlaDS)
    β     = 3 / 2    # 2nd turbulent flow exponent
    ρw    = 1000        # water density   [kg m^(-3)]
    ρi    = 910         # glacier density [kg m^(-3)]
    g     = 9.81         # gravitational acceleration [m s^(-2)]

    # water thickness - cavity opening (w)
    u_b = 1e-6       # ice sliding speed       [m/s]
    h_r = 0.1        # bedrock bump height     [m]
    l_r = 2.0        # bedrock bump wavelength [m]
    h_init = 0.05    # water sheet thickness

    # water thickness - cavity closing (v)
    Ã       = 2.5e-25 # ice rheological const
    glens_n = 3       # Glen's n

    # arrays
    # E-grid as unified C grids (c for main grid on center and v is the overlapped one)
    ∂ₓϕ  = (c=zeros(nx - 1, ny), v=zeros(nx, ny - 1))
    ∂ᵧϕ  = (c=zeros(nx, ny - 1), v=zeros(nx - 1, ny))
    qx   = (c=zeros(nx + 1, ny), v=zeros(nx, ny - 1))
    qy   = (c=zeros(nx, ny + 1), v=zeros(nx - 1, ny))
    ϕ    = (c=zeros(nx, ny), v=zeros(nx - 1, ny - 1))
    r    = (c=zeros(nx, ny), v=zeros(nx - 1, ny - 1))
    z    = (c=zeros(nx, ny), v=zeros(nx - 1, ny - 1))
    z0   = (c=zeros(nx, ny), v=zeros(nx - 1, ny - 1))
    s    = (c=zeros(nx, ny), v=zeros(nx - 1, ny - 1))
    P    = (c=zeros(nx, ny), v=zeros(nx - 1, ny - 1))
    D    = (vx=zeros(nx - 1, ny), vy=zeros(nx, ny - 1)) # D.vx for ∂ₓϕc, ∂ᵧϕv on Vx node of main grid; D.vy for ∂ᵧϕc, ∂ₓϕv on Vy node
    H    = (c=zeros(nx, ny), v=zeros(nx - 1, ny - 1))
    ρigH = (c=zeros(nx, ny), v=zeros(nx - 1, ny - 1))
    h    = (c=h_init * ones(nx, ny), v=h_init * ones(nx - 1, ny - 1)) # Water sheet thickness (shall compute initial thickness)

    # Geometry
    z_b = 0.0  # bed elevation [m]
    compute_H!(x) = 6.0 * (sqrt(x + 5000.0) - sqrt(5000.0)) + 1.0 - z_b # Ice thickness function (1D, varies only with x)
    xc = LinRange(0, lx, nx)
    xv = LinRange(0.5 * dx, lx - 0.5 * dx, nx - 1)
    for j in 1:ny
        @. H.c[:, j] = compute_H!(xc)
    end

    for j in 1:ny-1
        @. H.v[:, j] = compute_H!(xv)
    end

    @. ρigH.c = ρi * g * H.c
    @. ρigH.v = ρi * g * H.v

    # preconditioner and shadow arrays
    e = map(zero, ϕ)
    r̄ = map(zero, r)
    ϕ̄ = map(zero, ϕ)
    ∂ₓϕ̄ = map(zero, ∂ₓϕ)
    ∂ᵧϕ̄ = map(zero, ∂ᵧϕ)
    qx̄ = map(zero, qx)
    qȳ = map(zero, qy)
    D̄ = map(zero, D)
    ρigH̄ = map(zero, ρigH)
    h̄ = map(zero, h)
    Ic, Iv = coloring(nx, ny)

    # visualization
    do_visu = true
    if do_visu
        xv, yv = LinRange(dx / 2, lx - dx / 2, nx + 1), LinRange(dy / 2, ly - dy / 2, ny + 1)
        xc, yc = av(xv), av(yv)
        fig = Figure(; size=(900, 500))
        axs = [Axis(fig[1, 1]; xlabel="x", ylabel="ϕ"),
               Axis(fig[1, 2]; xlabel="Iterations / nx", ylabel="Relative Error", yscale=log10)]
        lns = lines!(axs[1], xc, ϕ.c[:, Int(ny / 2)])
    end

    # Time evolution
    s_d = 24.0 * 3600.0 # day           [s]
    t_tot = 5 * s_d     # total time    [s]
    @show Δt = 0.1 * s_d  # time stepsize [s]
    t = 0.0         # current time  [s]

    while t < t_tot
        # start of solve_ϕ!
        # Convergence tracking
        apt_iters = []
        apt_errs  = []
        τ         = 0.0   # [s] for monitoring conservation
        Δτ        = 1.0
        damp      = 0.0

        # monitor conservation
        dA = dx * dy
        ∫φ_0dA = (sum(ϕ.c) + sum(ϕ.v)) * dA / 2

        @timeit to "APT iterative loop" for iter in 1:maxiter
            # update solution
            @. ϕ.c += Δτ * s.c
            @. ϕ.v += Δτ * s.v

            if do_autotune && (iter % n_autotune == 0)
                @. z0.c = z.c
                @. z0.v = z.v
            end

            # compute residual
            @timeit to "residual!" residual!(r, ϕ, ∂ₓϕ, ∂ᵧϕ, qx, qy, D, ρigH, h, k, α, β, u_b, h_r, l_r, Ã, glens_n, dx, dy, m)

            # apply preconditioner to the residual
            if iter < 10
                @timeit to "Preconditioner" begin
                    # diagonal of J_CC
                    ɛ = 1e-9 # avoids division by zero
                    for group in Ic
                        fill!(e.c, 0.0)
                        e.c[group] .= 1.0
                        jvp!(e, r̄, r, ϕ, ϕ̄, ∂ₓϕ, ∂ₓϕ̄, ∂ᵧϕ, ∂ᵧϕ̄, qx, qx̄, qy, qȳ, D, D̄, ρigH, ρigH̄, h, h̄, k, α, β, u_b, h_r, l_r, Ã, glens_n, dx, dy,
                             m)
                        @. P.c[group] = 1.0 / (abs(r̄.c[group]) + ɛ)
                    end
    
                    # diagonal of J_VV
                    for group in Iv
                        fill!(e.v, 0.0)
                        e.v[group] .= 1.0
                        jvp!(e, r̄, r, ϕ, ϕ̄, ∂ₓϕ, ∂ₓϕ̄, ∂ᵧϕ, ∂ᵧϕ̄, qx, qx̄, qy, qȳ, D, D̄, ρigH, ρigH̄, h, h̄, k, α, β, u_b, h_r, l_r, Ã, glens_n, dx, dy,
                             m)
                        @. P.v[group] = 1.0 / (abs(r̄.v[group]) + ɛ)
                    end
                end
            end

            @. r.v[1, :]   = 0.0
            @. r.v[end, :] = r.v[end-1, :]
            @. r.v[:, 1]   = r.v[:, 2]
            @. r.v[:, end] = r.v[:, end-1]

            @. z.c = P.c * r.c
            @. z.v = P.v * r.v

            if do_autotune && (iter % n_autotune == 0)
                # compute A = abs(dot(s, (z - z0))) / dot(s, s)
                A = abs(sum(s.c .* (z.c - z0.c)) +
                        sum(s.v .* (z.v - z0.v))) / (sum(s.c .* s.c) +
                                                     sum(s.v .* s.v))

                damp = 1 + A - 2 * sqrt(A)
            end

            # compute damped search direction
            @. s.c = damp * s.c + z.c
            @. s.v = damp * s.v + z.v

            # monitor convergence
            if iter % ncheck == 0
                err = maximum(abs.(r.v)) / abs(m)
                err_rel = Δτ * maximum(abs, s.c) / maximum(abs, ϕ.c) # cheaper because no temp array
                push!(apt_iters, iter / nx)
                push!(apt_errs, err_rel)

                # checking conservation with quadrature approximation
                #  d/dt ∫ φ dA = m |Ω|.
                #  - divide by two on each cell
                #  - source term m counted doubly ∫φ_tdA ≈ (Σij φᶜ + Σij φᵛ) ΔA / 2
                ∫φ_tdA = (sum(ϕ.c) + sum(ϕ.v)) * dA / 2
                @printf("   iter = %.1f × nx, err = %1.3e, err_rel = %1.3e, ΔM = %1.3e\n", iter / nx, err, err_rel, (∫φ_tdA - ∫φ_0dA) - m * lx * ly * τ)

                if err_rel < etol
                    break
                end
            end

            # advance pseudo time for convergence check
            τ += Δτ
        end

        # update water sheet thickness array h
        h_prev = deepcopy(h)

        condc = 1.0 .* (h_prev.c .- h_r .< 1e-10)  # 1 where h_prevc < h_r, else 0    
        condv = 1.0 .* (h_prev.v .- h_r .< 1e-10)  # 1 where h_prevv < h_r, else 0

        @. h.c = (h_prev.c + Δt * (u_b / l_r) * h_r * condc) / (1 + Δt * (Ã * abs(ρigH.c - ϕ.c)^(glens_n - 1) * abs(ρigH.c - ϕ.c) + (u_b / l_r) * condc))
        @. h.v = (h_prev.v + Δt * (u_b / l_r) * h_r * condv) / (1 + Δt * (Ã * abs(ρigH.v - ϕ.v)^(glens_n - 1) * abs(ρigH.v - ϕ.v) + (u_b / l_r) * condv))

        # advance time
        t += Δt
        @printf("Current Time in days: %.2f\n", round(t / s_d, digits=2))  # Show time in days
    end

    # plot
    if do_visu
        fig = Figure(; size=(1000, 500))

        # Choose slice indices for plotting
        y_slice_idx = Int(ny ÷ 2)  # middle slice in y-direction
        xc_slice = LinRange(dx / 2, lx - dx / 2, nx)

        # Create axis for ice thickness H (slice through middle in y-direction)
        ax_H = Axis(fig[1, 1];
                    xlabel="Distance x [m]",
                    ylabel="Ice Thickness H [m]",
                    title="Ice Thickness (y-slice at y=$(yc[y_slice_idx]))")
        lines!(ax_H, xc_slice, H.c[:, y_slice_idx]; linewidth=2, color=:blue, label="Ice Thickness")

        # Create axis for effective pressure (slice through middle in y-direction)
        ax_N = Axis(fig[2, 1];
                    xlabel="Distance x [m]",
                    ylabel="Effective Pressure N [Pa]",
                    title="Effective Pressure (t = $(round(t/s_d, digits=2)) days)")
        lines!(ax_N, xc_slice, abs.(ρigH.c - ϕ.c)[:, y_slice_idx]; linewidth=2, color=:green, label="N")

        # Create axis for hydraulic potential (slice through middle in y-direction)
        ax_ϕ = Axis(fig[1, 2];
                    xlabel="Distance x [m]",
                    ylabel="Hydraulic Potential ϕ [Pa]",
                    title="Hydraulic Potential (t = $(round(t/s_d, digits=2)) days)")
        lines!(ax_ϕ, xc_slice, ϕ.c[:, y_slice_idx]; linewidth=2, color=:orange, label="ϕ")

        # Create axis for water sheet thickness (slice through middle in y-direction)
        ax_h = Axis(fig[2, 2];
                    xlabel="Distance x [m]",
                    ylabel="Water Sheet Thickness h [m]",
                    title="Water Sheet Thickness (t = $(round(t/s_d, digits=2)) days)")
        lines!(ax_h, xc_slice, h.c[:, y_slice_idx]; linewidth=2, color=:cornflowerblue, label="h")

        # Plot reference points (assuming they are 1D along x-direction)
        include("check-vs-gladsog.jl")

        # Get data arrays at melting parameter m
        ref_phi, ref_h = dataOG[m]

        # Find the closest reference time point
        if isapprox(t, 0; atol=Δt)
            t_idx = 1
        elseif isapprox(t, 5 * s_d; atol=Δt)
            t_idx = 2
        elseif isapprox(t, 1000 * s_d; atol=Δt)
            t_idx = 3
        else
            # Find the closest reference time point
            ref_times = [0, 5 * s_d, 1000 * s_d]
            _, t_idx = findmin(abs.(ref_times .- t))
            @warn "No exact match for time t=$(t/s_d) days. Using closest reference time: $(ref_times[t_idx]/s_d) days."
        end

        # Styling
        ref_colors  = [:darkblue, :darkred, :darkgreen]
        markers     = [:circle, :utriangle, :diamond]
        time_labels = ["Initial (t=0)", "Transient (t=5 days)", "Steady (t=1000 days)"]
        out_x       = 1e3 * [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

        scatter!(ax_ϕ, out_x, ref_phi[:, t_idx];
                 color=ref_colors[t_idx], marker=markers[t_idx], markersize=12,
                 strokewidth=1, strokecolor=:black,
                 label="Ref: " * time_labels[t_idx])

        scatter!(ax_h, out_x, ref_h[:, t_idx];
                 color=ref_colors[t_idx], marker=markers[t_idx], markersize=12,
                 strokewidth=1, strokecolor=:black,
                 label="Ref: " * time_labels[t_idx])

        # Add legends
        axislegend(ax_H; position=:rt)
        axislegend(ax_N; position=:rt)
        axislegend(ax_ϕ; position=:rt)
        axislegend(ax_h; position=:rt)

        # Update layout and display
        fig[0, :] = Label(fig, "SHMIP Suite A (2D E-grid, y-slice)\nm = $m"; fontsize=20)
        display(fig)

        # save the figure
        # save("shmip_suite_a_2d_m$(m).png", fig)
    end

    show(to)
end

sheet_2d_e_grid()