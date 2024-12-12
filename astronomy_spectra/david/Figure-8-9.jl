### A Pluto.jl notebook ###
# v0.17.4

using Markdown
using InteractiveUtils

# ╔═╡ 61dc49c3-0709-4e80-8e4e-ed3b6ee45d01
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ c36c9649-bbd0-4c42-9391-ab81df67967d
using CacheVariables, CairoMakie, DataFrames, DelimitedFiles, FITSIO, HTTP, Interpolations, LinearAlgebra, Printf, ProgressLogging, Statistics, StructArrays

# ╔═╡ 41f9836d-239a-4773-9ede-931c01b29edb
using Logging, TerminalLoggers; global_logger(TerminalLogger());

# ╔═╡ bdc1473e-6cbe-11ec-3f1d-ebcb2ac8a4c2
md"""
# Figures 8-9
"""

# ╔═╡ dbda889d-53fb-4f6a-91e5-8624854146c8
md"""
## Setup
"""

# ╔═╡ f2d57fab-23be-4e7b-ba23-09b66470875a
WAVE = 1480:0.5:1620

# ╔═╡ 76452b0a-e7df-4788-8199-4f28249be5b1
k = 5

# ╔═╡ 8f4b670a-b2fb-4401-802e-d8e64f6959e0
CONFIGSTR = "wave-$(replace(string(WAVE),':'=>'-')),k-$k"

# ╔═╡ 2dd96a13-7dd2-4801-b41b-ad871e98acef
md"""
## Load full data
"""

# ╔═╡ b77a4927-afec-4162-9224-523f62ea78a5
function cacheurl(url; dir="data", update_period=1)
	ispath(dir) || mkpath(dir)
	
	path = joinpath(dir, basename(url))
	ispath(path) || try
		HTTP.download(url, path; update_period)
	catch e
		ispath(path) && rm(path)
		throw(e)
	end
	
	return path
end

# ╔═╡ 37e1de1f-c9bd-4538-82cf-38f2e5b95efc
function loadspec(SURVEY, RUN2D, PLATE, MJD, FIBERID; dir=joinpath("data","specs"))
	url = string(
		"https://dr16.sdss.org/sas/dr16/",
		SURVEY,
		"/spectro/redux/",
		RUN2D,
		"/spectra/lite/",
		@sprintf("%04d", PLATE),
		"/spec-",
		@sprintf("%04d", PLATE),
		"-",
		MJD,
		"-",
		@sprintf("%04d", FIBERID),
		".fits"
	)
	return FITS(f -> DataFrame(f[2]), cacheurl(url; dir, update_period=Inf))
end

# ╔═╡ 91149d59-bad7-4586-855e-eef4fa2dc3e8
Yfull, vfull = cache(joinpath("cache","data,$CONFIGSTR.bson")) do
	# Load DR16Q and form list of spectra to download
	@info "Load DR16Q and form list of spectra to download"
	DR16Q_URL = "https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/DR16Q_v4.fits"
	PLATE_URL = "https://data.sdss.org/sas/dr16/sdss/spectro/redux/plates-dr16.fits"
	dr16qp = innerjoin(
		FITS(f -> DataFrame(f[2]), cacheurl(DR16Q_URL)),
		FITS(f -> DataFrame(f[2]), cacheurl(PLATE_URL));
		on=[:PLATE,:MJD], validate=(false,true), makeunique=true
	)
	list = filter(dr16qp) do spec
		spec.SURVEY == "eboss" && spec.PLATEQUALITY == "good" &&
		2.0 < spec.Z < 2.1 && spec.BAL_PROB < 0.2
	end

	# Load spectra and filter based on wavelengths covered
	@info "Load spectra and filter based on wavelengths covered"
	specs = @withprogress name="Loading spectra" map(
		1:nrow(list),
		list.SURVEY, list.RUN2D, list.PLATE, list.MJD, list.FIBERID, list.Z
	) do idx, SURVEY, RUN2D, PLATE, MJD, FIBERID, Z
		df = loadspec(SURVEY,RUN2D,PLATE,MJD,FIBERID)
		spec = (;
			WAVE = 10.0 .^ df.LOGLAM ./ (1+Z),   # Restframe wavelength
			FLUX = df.FLUX,                      # Flux
			IVAR = df.IVAR,                      # Inverse variance
		)
		@logprogress idx/nrow(list)
		return spec
	end
	specs = filter(specs) do spec
		first = findlast(<(minimum(WAVE)),spec.WAVE)
		last = findfirst(>(maximum(WAVE)),spec.WAVE)
		return !isnothing(first) && !isnothing(last) &&  # spec.WAVE covers WAVE
			!any(iszero,spec.IVAR[first:last])           # without missing entries
	end

	# Interpolate, center, normalize, and filter based on variance profiles
	@info "Interpolate, center, normalize, and filter based on variance profiles"
	normspecs = (StructArray∘map)(specs) do spec
		# Form data vectors via linear interpolation
		FLUX = LinearInterpolation(spec.WAVE, spec.FLUX).(WAVE)
		VAR  = LinearInterpolation(spec.WAVE, inv.(spec.IVAR)).(WAVE)

		# Center
		FLUX = FLUX .- mean(FLUX)

		# Normalize
		scaling = abs(mean(FLUX[1525 .< WAVE .< 1575]))
		FLUX = FLUX / scaling
		VAR  = VAR  / scaling^2
		
		return (;FLUX, VAR)
	end
	normspecs = filter(spec -> minimum(spec.VAR)/maximum(spec.VAR) > 0.4, normspecs)
	normspecs = sort(normspecs; by = spec->mean(spec.VAR))

	# Form data matrix and variance vector
	@info "Form data matrix and variance vector"
	flux = reduce(hcat,normspecs.FLUX)
	mvar = mean.(normspecs.VAR)

	return flux, mvar
end

# ╔═╡ 77a7a40e-43cd-412f-8504-a45bc3bc5f1a
md"""
## Compute and plot "ground-truth" components
"""

# ╔═╡ ef7eb005-71e2-4384-a836-634ae7e4f7f6
U = svd(Yfull[:,1:5000]).U[:,1:k]

# ╔═╡ 4bf838f1-ef88-475c-931a-41af6ae1580f
figU = with_theme(;linewidth=2) do
	fig = Figure(;resolution=(1050,210))

	for (i,ui) in enumerate(eachcol(U))
		flip = maximum(abs,filter(>(0),ui)) < maximum(abs,filter(<(0),ui))
		ax = Axis(fig[1,i]; yticks = [0.0], xticks = [1500,1600])
		lines!(ax, WAVE, flip ? -ui : ui; linewidth=2)
		tightlimits!(ax, Left(), Right())
	end
	linkxaxes!(contents(fig.layout)...)

	for i in 1:k
		Label(fig[1,i], L"\mathbf{u}_{%$i}";
			tellheight = false, tellwidth = false, halign = :left, valign = :top,
			padding = (8,0,0,-8), textsize = 24f0,
		)
	end
	Label(fig[end+1,:], "x-axis: rest frame wavelength, y-axis: component entries")

	fig
end

# ╔═╡ 6fac3e31-3d66-423b-9efb-4a3c28b8583d
md"""
## Form and plot test dataset
"""

# ╔═╡ 7273c18a-8d0b-4382-b96b-be91dfe6bfba
testidx = [1:3000; length(vfull)-2000+1:length(vfull)]

# ╔═╡ b3cc9d2f-ebaf-4e39-abb0-d05f233e810e
Y = Yfull[:,testidx]

# ╔═╡ d30a7e6f-4536-4c33-9ee1-2bf81ac13171
v = vfull[testidx]

# ╔═╡ 23a24450-8c1f-4560-8718-cd787c9873e6
n = length(v)

# ╔═╡ dfed364c-c6f5-476a-b20f-1883f5bc04f7
figYv = with_theme(;linewidth=2) do
	fig = Figure(;resolution=(540,300))

	axY = Axis(fig[1,1]; yreversed = true, ylabel = "Rest frame\nwavelength")
	hmY = heatmap!(axY, 1:n, WAVE, permutedims(Y);
		colormap = :vik, colorrange = (-1,1).*3.25)
	Colorbar(fig[1,2], hmY; label=L"\mathbf{Y}", labelsize=18)

	axv = Axis(fig[2,1];
		limits = (nothing, nothing, -0.25, 6), yticks = 0:3:6,
		xlabel = "Spectrum index", ylabel = "Variance",
	)
	lines!(axv, 1:n, v)

	axY.xticks = axv.xticks = 1000:1000:5000
	linkxaxes!(axY,axv)
	hidexdecorations!(axY; ticks=false)
	rowsize!(fig.layout, 2, Relative(0.2))
	rowgap!(fig.layout, 10)
	
	fig
end

# ╔═╡ 50da813d-0464-449f-986b-67350166b5d9
figspecs = with_theme(;linewidth=2) do
	fig = Figure(resolution=(486,300))
	ax = Axis(fig[1,1];
		xticks = 1500:50:1600,
		xlabel="Rest frame wavelength", ylabel="Spectrum (flux)"
	)

	for idx in [3380,2280,1]
		vstr = @sprintf("%0.2f", v[idx])
		lines!(ax, WAVE, Y[:,idx]; label=L"v_\ell \approx %$vstr")
	end
	tightlimits!(ax, Left(), Right())

	Legend(fig[0,1], ax;
		orientation = :horizontal, tellheight = true, tellwidth = false,
		framevisible = false, padding = (0,0,0,0),
	)
	rowgap!(fig.layout,10)
	fig
end

# ╔═╡ 76533f07-fd31-4c98-bd05-42bb1bcafacb
md"""
## Estimate signal variances
"""

# ╔═╡ 78368c22-dddf-4a6f-a5b5-6bf4a1a2e2e5
function λest_inv(Y, v)
	n, L = size.(Y,2), length(Y)
	w = [(1/v[l])/sum(n[lp]/v[lp] for lp in 1:L) for l in 1:L]
	
	Yw = reduce(hcat,sqrt(w[l])*Y[l] for l in 1:L)
	return svdvals(Yw).^2
end

# ╔═╡ 6167e3a6-742a-48d0-a8cb-a825e2c77a92
Ξ(λ, v; c) = -(v+v/c-λ)/2 + sqrt((v+v/c-λ)^2-4*v^2/c)/2

# ╔═╡ c703d3d7-32a9-4395-a030-e5ce2c581992
function λest(Y; v)
	d, n, L = (only∘unique)(size.(Y,1)), size.(Y,2), length(Y)
	p = n./sum(n)
	
	λinv = λest_inv(Y,v)
	c = sum(n)/d
	vb = inv(sum(p[l]/v[l] for l in 1:L))

	k = count(>(vb*(1+1/sqrt(c))^2), λinv)
	return [Ξ(λinv[i],vb; c) for i in 1:k]
end

# ╔═╡ e517af63-b2d1-4338-920d-a4ca30583b43
λh = λest(collect(reshape.(eachcol(Y),:,1)); v)

# ╔═╡ d7fa3c1c-5eb2-4414-8deb-44bbf7df70d1
md"""
### Evaluate weighted PCA methods
"""

# ╔═╡ 18e0c566-6620-4134-9c2a-1175b9523c35
wlist = (;
	optlim = [inv.(v.*(λh[i].+v)) for i in 1:k],
	invvar = [inv.(v) for i in 1:k],
	unweight = [fill(1,n) for i in 1:k],
)

# ╔═╡ 769174c2-ec78-46a8-87dd-577a423cece1
rlist = map(wlist) do w
	r = map(1:k) do i
		Yw = Y*sqrt(Diagonal(w[i]))
		uhi = svd(Yw).U[:,i]
		return abs2(U[:,i]'uhi)
	end
end

# ╔═╡ 96c59aea-5203-43c1-82c4-39c11166b4c8
md"""
**Rounded version:**
"""

# ╔═╡ d73b8ecb-53be-421f-9145-2aaa4a37dc39
map(r -> join([@sprintf("%0.3f",ri) for ri in r]," | "), rlist)

# ╔═╡ 13b4a803-5dc0-43ec-9ccc-e5a301b75adf
md"""
**Approximate Orthogonality:**
"""

# ╔═╡ 0f0093f5-5373-44f2-b6ae-d8f8ba67dc47
let w=wlist[:optlim]
	Uh = map(1:k) do i
		Yw = Y*sqrt(Diagonal(w[i]))
		return svd(Yw).U[:,i]
	end |> (uh -> reduce(hcat,uh))
	maximum(abs2(Uh[:,i]'Uh[:,j]) for i in 1:k for j in 1:k if i != j)
end

# ╔═╡ 9a577acf-7bb4-43b1-8c61-1c9de62468e9
md"""
## Save figure and data to files
"""

# ╔═╡ b89eee8c-fe66-4ba5-ba99-5f468ed7fdda
!ispath("outs") && mkpath("outs");

# ╔═╡ 611ea929-fe23-4cc7-abe0-1995d6170a7a
save(joinpath("outs","sdss,U.png"), figU)

# ╔═╡ 7282f488-6284-4b29-9615-cc653b272799
save(joinpath("outs","sdss,data.png"), figYv)

# ╔═╡ 5a14ba22-4a14-4c07-8928-cba42e8adf3b
save(joinpath("outs","sdss,specs.png"), figspecs)

# ╔═╡ 5d3721a1-7ba0-4113-9f21-22bcfce73f3e
writedlm(
	joinpath("outs","sdss,rec,$CONFIGSTR.dat"),
	reduce(hcat,vcat(string(method),r) for (method,r) in pairs(rlist))
)

# ╔═╡ 62458ab8-c540-4a23-8622-8086eff5bca7
writedlm(
	joinpath("outs","sdss,rec,$CONFIGSTR,rounded.dat"),
	reduce(hcat,
		vcat(string(method),[@sprintf("%0.3f",ri) for ri in r])
		for (method,r) in pairs(rlist)
	)
)

# ╔═╡ Cell order:
# ╟─bdc1473e-6cbe-11ec-3f1d-ebcb2ac8a4c2
# ╠═61dc49c3-0709-4e80-8e4e-ed3b6ee45d01
# ╠═c36c9649-bbd0-4c42-9391-ab81df67967d
# ╠═41f9836d-239a-4773-9ede-931c01b29edb
# ╟─dbda889d-53fb-4f6a-91e5-8624854146c8
# ╠═f2d57fab-23be-4e7b-ba23-09b66470875a
# ╠═76452b0a-e7df-4788-8199-4f28249be5b1
# ╠═8f4b670a-b2fb-4401-802e-d8e64f6959e0
# ╟─2dd96a13-7dd2-4801-b41b-ad871e98acef
# ╠═b77a4927-afec-4162-9224-523f62ea78a5
# ╠═37e1de1f-c9bd-4538-82cf-38f2e5b95efc
# ╠═91149d59-bad7-4586-855e-eef4fa2dc3e8
# ╟─77a7a40e-43cd-412f-8504-a45bc3bc5f1a
# ╠═ef7eb005-71e2-4384-a836-634ae7e4f7f6
# ╠═4bf838f1-ef88-475c-931a-41af6ae1580f
# ╟─6fac3e31-3d66-423b-9efb-4a3c28b8583d
# ╠═7273c18a-8d0b-4382-b96b-be91dfe6bfba
# ╠═b3cc9d2f-ebaf-4e39-abb0-d05f233e810e
# ╠═d30a7e6f-4536-4c33-9ee1-2bf81ac13171
# ╠═23a24450-8c1f-4560-8718-cd787c9873e6
# ╠═dfed364c-c6f5-476a-b20f-1883f5bc04f7
# ╠═50da813d-0464-449f-986b-67350166b5d9
# ╟─76533f07-fd31-4c98-bd05-42bb1bcafacb
# ╠═78368c22-dddf-4a6f-a5b5-6bf4a1a2e2e5
# ╠═6167e3a6-742a-48d0-a8cb-a825e2c77a92
# ╠═c703d3d7-32a9-4395-a030-e5ce2c581992
# ╠═e517af63-b2d1-4338-920d-a4ca30583b43
# ╟─d7fa3c1c-5eb2-4414-8deb-44bbf7df70d1
# ╠═18e0c566-6620-4134-9c2a-1175b9523c35
# ╠═769174c2-ec78-46a8-87dd-577a423cece1
# ╟─96c59aea-5203-43c1-82c4-39c11166b4c8
# ╠═d73b8ecb-53be-421f-9145-2aaa4a37dc39
# ╟─13b4a803-5dc0-43ec-9ccc-e5a301b75adf
# ╠═0f0093f5-5373-44f2-b6ae-d8f8ba67dc47
# ╟─9a577acf-7bb4-43b1-8c61-1c9de62468e9
# ╠═b89eee8c-fe66-4ba5-ba99-5f468ed7fdda
# ╠═611ea929-fe23-4cc7-abe0-1995d6170a7a
# ╠═7282f488-6284-4b29-9615-cc653b272799
# ╠═5a14ba22-4a14-4c07-8928-cba42e8adf3b
# ╠═5d3721a1-7ba0-4113-9f21-22bcfce73f3e
# ╠═62458ab8-c540-4a23-8622-8086eff5bca7
