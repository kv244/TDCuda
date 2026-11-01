"""
build_particle_surface.py  --  add the isosurface ("Option B") to the particle cloud
=======================================================================
Augments an EXISTING /particle_cloud_cuda component (built by
build_particle_cloud_cuda.py) with a live watertight surface mesh:

  cuda_particles (ParticleCloudTOP, CUDA)       -> RGBA32F positions, alpha 0 = padding
        |  wired input 0
  density_top    (ParticleDensityTOP, CUDA)     -> splat + blur on the GPU, R32F slice atlas
        |  TOP to CHOP  (Download Type: next frame -> no CPU/GPU stall)
  density_chop                                   -> one channel, W*H samples
        |  'Density CHOP' parameter
  surface_geo / surface  (ParticleSurfaceSOP)    -> marching tetrahedra -> welded mesh
        |
  surface_mat (Phong, emissive) -> added to render1's geometry list

A new "Surface" custom page on the component drives everything. Gridres and
Boundssize are bound to BOTH the density TOP and the SOP, which is what keeps
the atlas layout and grid geometry in agreement (see DensityAtlas.h).

RUN: build_particle_cloud_cuda.py first, then paste this into a Text DAT ->
right-click -> Run Script. Re-running is safe: it removes what it added before
and rebuilds (the base cloud is untouched).
"""

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BUILD_DIR    = r"C:/Users/julia/Downloads/TDClaude/build/Release"
DENSITY_DLL  = BUILD_DIR + "/ParticleDensityTOP.dll"
SURFACE_DLL  = BUILD_DIR + "/ParticleSurfaceSOP.dll"

CONTAINER    = "/particle_cloud_cuda"

SURFACE_ON   = True            # show the mesh
SPRITES_ON   = True            # keep the glowing sprites too
GRID_RES     = 64              # 64 ~ 2 ms/cook on the CPU side; 96 ~ 5 ms; 128 ~ 12 ms
BOUNDS_SIZE  = 32.0            # world-space cube (centred on origin) the grid covers
SMOOTH       = 0.6             # density blur radius (world units)
ISO          = 4.0             # surface at "N x mean density"; raise to tighten
SURF_RGB     = (0.05, 0.55, 0.9)

ADDED_OPS    = ('density_top', 'density_chop', 'surface_geo', 'surface_mat')
PAGE_NAME    = 'Surface'


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _note(msg):
    print("[surface] " + msg)


def _set_first(owner, names, value, what):
    """Set whichever of several candidate parameter names exists."""
    for n in names:
        if hasattr(owner.par, n):
            try:
                setattr(owner.par, n, value)
                return True
            except Exception as e:
                _note("%s via '%s' failed (%s)" % (what, n, e))
    _note("none of %s found for %s" % (names, what))
    return False


def _set_menu(owner, names, wanted_substrings, what):
    """Set a menu parameter by matching its REAL menu names against substrings,
    so we don't depend on exact token spellings that vary between builds."""
    for n in names:
        if not hasattr(owner.par, n):
            continue
        p = getattr(owner.par, n)
        try:
            options = list(p.menuNames)
        except Exception:
            continue
        for want in wanted_substrings:
            for opt in options:
                if want in opt.lower():
                    p.val = opt
                    _note("%s = %s" % (what, opt))
                    return True
        _note("%s: no option matching %s in %s" % (what, wanted_substrings, options))
        return False
    _note("none of %s found for %s" % (names, what))
    return False


def _bind(target, par_name, comp_par):
    if not hasattr(target.par, par_name):
        _note("%s has no parameter '%s' (plugin not loaded?)" % (target.path, par_name))
        return
    p = getattr(target.par, par_name)
    p.mode = ParMode.BIND
    p.bindExpr = "op('%s').par.%s" % (CONTAINER, comp_par)


def _load_plugin(o, dll):
    o.par.plugin = dll
    try:
        o.cook(force=True)          # custom params appear once the DLL has loaded
    except Exception:
        pass


# ----------------------------------------------------------------------
def build():
    box = op(CONTAINER)
    if box is None:
        raise RuntimeError("Run build_particle_cloud_cuda.py first: %s not found" % CONTAINER)
    cuda = box.op('cuda_particles')
    render = box.op('render1')

    # --- idempotent: remove anything a previous run added ----------------
    for name in ADDED_OPS:
        if box.op(name):
            box.op(name).destroy()
    for page in box.customPages:
        if page.name == PAGE_NAME:
            page.destroy()

    # --- 1. density TOP (GPU splat + blur), fed by the position TOP ------
    dens = box.create(cplusplusTOP, 'density_top')
    dens.nodeX, dens.nodeY = cuda.nodeX + 250, cuda.nodeY - 350
    dens.inputConnectors[0].connect(cuda)
    _load_plugin(dens, DENSITY_DLL)

    # --- 2. TOP to CHOP: download only the density grid ------------------
    chop = box.create(toptoCHOP, 'density_chop')
    chop.nodeX, chop.nodeY = dens.nodeX + 250, dens.nodeY
    _set_first(chop, ['top'], dens.path, 'TOP to CHOP source')
    # next-frame download: the CPU reads last frame's grid instead of stalling
    # until the GPU finishes this one (surface lags the sprites by one frame)
    _set_menu(chop, ['downloadtype'], ['next', 'delay'], 'download type')
    _set_menu(chop, ['crop'], ['full'], 'crop')
    # The whole image as ONE set of channels (W*H samples each, row-major from
    # the bottom row) instead of the default one-channel-per-row layout.
    _set_first(chop, ['singleset'], True, 'single set (one channel per colour)')
    # The density is DATA, not colour: no transfer curve may touch it. The
    # default output colour space (sRGB) would gamma-warp every density value.
    _set_menu(chop, ['outputcolorspace'], ['passthrough'], 'output colour space')
    # r/g/b/a are channel NAMES (blank = channel omitted). The atlas is mono,
    # so keep only red, named 'density' -- the SOP reads channel 0.
    for ch, name in (('r', 'density'), ('g', ''), ('b', ''), ('a', '')):
        if hasattr(chop.par, ch):
            setattr(chop.par, ch, name)

    # --- 3. surface material ---------------------------------------------
    mat = box.create(phongMAT, 'surface_mat')
    mat.nodeX, mat.nodeY = dens.nodeX + 500, dens.nodeY - 150
    _set_first(mat, ['cullface'], 'neither', 'two-sided (no culling)')

    # --- 4. surface geometry with the custom SOP --------------------------
    geo = box.create(geometryCOMP, 'surface_geo')
    geo.nodeX, geo.nodeY = dens.nodeX + 750, dens.nodeY
    for child in list(geo.children):
        child.destroy()
    sop = geo.create(cplusplusSOP, 'surface')
    _load_plugin(sop, SURFACE_DLL)
    sop.render = True            # freshly created SOPs have render/display OFF
    sop.display = True           # (same gotcha that blanked the billboards)
    if hasattr(sop.par, 'Density'):
        sop.par.Density = chop.path
    else:
        _note("SOP has no 'Density' parameter yet -- check SURFACE_DLL path")
    geo.par.material = mat.path

    # --- 5. render it alongside the sprites --------------------------------
    # use the raw string the base builder typed ('<cloud> <cloud_tri>'), not
    # eval(), which can return OP objects for an operator-list parameter
    paths = str(render.par.geometry.val)
    if geo.path not in paths.split():
        render.par.geometry = (paths + ' ' + geo.path).strip()

    # --- 6. the "Surface" control page -------------------------------------
    pg = box.appendCustomPage(PAGE_NAME)
    pg.appendToggle('Surface', label='Show Surface')[0].val = SURFACE_ON
    pg.appendToggle('Sprites', label='Show Sprites')[0].val = SPRITES_ON
    pg.appendInt('Gridres',    label='Grid Resolution')[0].val = GRID_RES
    pg.appendFloat('Boundssize', label='Bounds Size')[0].val   = BOUNDS_SIZE
    pg.appendFloat('Smooth',   label='Smoothing')[0].val       = SMOOTH
    pg.appendFloat('Iso',      label='Isolevel')[0].val        = ISO
    col = pg.appendRGB('Surfacecolor', label='Surface Color')
    col[0].val, col[1].val, col[2].val = SURF_RGB
    pg.appendToggle('Flipnormals', label='Flip Normals')[0].val = False
    pg.appendToggle('Flipatlasy',  label='Flip Atlas Rows')[0].val = False
    for nm, (lo, hi) in {'Gridres': (16, 160), 'Boundssize': (4, 120),
                         'Smooth': (0, 3), 'Iso': (0.5, 50)}.items():
        p = getattr(box.par, nm)
        p.normMin, p.normMax = lo, hi

    # Gridres + Boundssize MUST match on both ops -> bind both to one control
    for nm in ('Gridres', 'Boundssize', 'Smooth'):
        _bind(dens, nm, nm)
    for nm in ('Gridres', 'Boundssize', 'Iso', 'Flipnormals', 'Flipatlasy'):
        _bind(sop, nm, nm)

    # material colour (diffuse + a little emission so it reads with the bloom)
    for comp_c, (d_par, e_par) in zip(('Surfacecolorr', 'Surfacecolorg', 'Surfacecolorb'),
                                      (('diffr', 'emitr'), ('diffg', 'emitg'), ('diffb', 'emitb'))):
        for par_name, scale in ((d_par, 1.0), (e_par, 0.35)):
            if hasattr(mat.par, par_name):
                getattr(mat.par, par_name).expr = "op('%s').par.%s * %s" % (CONTAINER, comp_c, scale)

    # visibility toggles: the surface geo, and the sprite geos keep their
    # Render Mode logic but also obey 'Show Sprites'
    geo.par.render.expr = "op('%s').par.Surface.eval()" % CONTAINER
    for name, mode in (('cloud', 'billboard'), ('cloud_tri', 'triangle')):
        g = box.op(name)
        if g is not None:
            g.par.render.expr = ("op('%s').par.Sprites.eval() and "
                                 "op('%s').par.Rendermode.eval() == '%s'" % (CONTAINER, CONTAINER, mode))

    _note("built: density_top -> density_chop -> surface_geo/surface. "
          "Controls on the '%s' page. Tune Isolevel live." % PAGE_NAME)
    return geo


build()
