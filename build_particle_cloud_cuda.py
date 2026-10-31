"""
build_particle_cloud_cuda.py  --  billboard sprites + stateful sim + exposed UI
=======================================================================
Builds the CUDA particle cloud in TouchDesigner as a self-contained,
GUI-controllable component.

ParticleCloudTOP.dll (CUDA, stateful) integrates position+velocity every
frame and outputs an RGBA32F position texture. A GLSL MAT reads that texture
in the vertex shader (texelFetch by TDInstanceID) and draws each particle as
a small camera-facing QUAD (billboard) sized in the vertex shader. Two
triangles per particle scales to hundreds of thousands of particles.

The /particle_cloud_cuda COMP gets its own Custom Parameter page ("Particle
Cloud"), so every control is adjustable from the component's parameter dialog
at runtime. The sim params bind to the CUDA TOP, camera/bloom bind to those
operators, and Color/Sprite Size drive GLSL uniforms (uColor / uSprite).

RUN: paste into a Text DAT -> right-click -> Run Script. View the 'out' TOP.
Re-running rebuilds /particle_cloud_cuda cleanly.
"""

# ----------------------------------------------------------------------
# CONFIG  (these become the DEFAULTS of the component's custom parameters)
# ----------------------------------------------------------------------
DLL_PATH      = r"C:/Users/julia/Downloads/TDClaude/build/Release/ParticleCloudTOP.dll"

NUM_PARTICLES = 200000         # billboards are cheap -> push this high
CLOUD_RADIUS  = 6.0
SIM_SPEED     = 1.0
GRAVITY       = 0.8
SWIRL         = 1.4
DAMPING       = 0.985
TURBULENCE    = 0.4

SPRITE_SIZE   = 0.10           # world-space size of each sprite
GLOW          = 0.06           # per-particle intensity (additive accumulates it)
CAM_DIST      = 38.0           # pulled back so the cloud reads as a glowing orb
NEON_RGB      = (0.0, 0.92, 1.0)
BLOOM_SIZE    = 10.0

CONTAINER_NAME = "particle_cloud_cuda"

# ----------------------------------------------------------------------
# SHADERS  (color + sprite size come from GLSL uniforms so they are
#           controllable at runtime; uColor.rgb and uSprite.x are fed by
#           the GLSL MAT's vector uniforms, bound to the component params)
# ----------------------------------------------------------------------
VERTEX_SHADER = """
uniform sampler2D sPos;
uniform vec4 uSprite;      // .x = world-space sprite size
out vec2 vUV;

void main()
{
    int id  = TDInstanceID();
    int w   = textureSize(sPos, 0).x;
    ivec2 t = ivec2(id % w, id / w);
    vec3 particlePos = texelFetch(sPos, t, 0).xyz;

    vUV = P.xy + 0.5;                  // 0..1 across the sprite quad
    vec3 local = P * uSprite.x;        // scale local quad to world size
    vec4 world = TDDeform(vec4(local + particlePos, 1.0));
    gl_Position = TDWorldToProj(world);
}
"""

PIXEL_SHADER = """
uniform vec4 uColor;       // .rgb = particle color, .a = glow intensity
out vec4 fragColor;
in vec2 vUV;

void main()
{
    vec2 d = vUV * 2.0 - 1.0;          // round, soft-edged sprite
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float a = smoothstep(1.0, 0.0, r2) * uColor.a;   // dim per particle so
    fragColor = TDOutputSwizzle(vec4(uColor.rgb * a, a));  // additive accumulates
}
"""

# ----------------------------------------------------------------------
def _set_first(owner, names, value, what):
    for n in names:
        if hasattr(owner.par, n):
            try:
                setattr(owner.par, n, value); return True
            except Exception as e:
                print(f"[cuda cloud] note: {what} via '{n}' failed ({e})")
    print(f"[cuda cloud] note: none of {names} found for {what}")
    return False


def _expose(box, cuda, geo, mat, cam, blur):
    """Add a Custom Parameter page to the COMP and wire every control to it."""
    pg = box.appendCustomPage('Particle Cloud')

    # --- simulation controls (bind -> CUDA TOP) ---
    pg.appendInt('Count',      label='Particle Count')[0].val = NUM_PARTICLES
    pg.appendFloat('Radius',   label='Cloud Radius')[0].val   = CLOUD_RADIUS
    pg.appendFloat('Speed')[0].val                            = SIM_SPEED
    pg.appendFloat('Gravity',  label='Gravity Well')[0].val   = GRAVITY
    pg.appendFloat('Swirl')[0].val                            = SWIRL
    pg.appendFloat('Damping')[0].val                          = DAMPING
    pg.appendFloat('Turbulence')[0].val                       = TURBULENCE
    pg.appendPulse('Reset')

    # --- look controls ---
    col = pg.appendRGB('Color')
    col[0].val, col[1].val, col[2].val = NEON_RGB
    pg.appendFloat('Spritesize', label='Sprite Size')[0].val    = SPRITE_SIZE
    pg.appendFloat('Glow',       label='Glow Intensity')[0].val = GLOW
    pg.appendFloat('Camdist',    label='Camera Distance')[0].val = CAM_DIST
    pg.appendFloat('Bloom')[0].val                               = BLOOM_SIZE

    # nice slider ranges
    ranges = {'Count': (1024, 1048576), 'Radius': (0.1, 50), 'Speed': (0, 5),
              'Gravity': (0, 5), 'Swirl': (0, 8), 'Damping': (0.9, 1.0),
              'Turbulence': (0, 5), 'Spritesize': (0.01, 1.0), 'Glow': (0, 1),
              'Camdist': (2, 80), 'Bloom': (0, 30)}
    for nm, (lo, hi) in ranges.items():
        p = getattr(box.par, nm)
        p.normMin, p.normMax = lo, hi

    def bind(target_op, par_name, comp_par):
        pr = getattr(target_op.par, par_name)
        pr.mode = ParMode.BIND
        pr.bindExpr = "parent().par.%s" % comp_par

    # simulation params -> CUDA TOP
    for nm in ('Count', 'Radius', 'Speed', 'Gravity', 'Swirl', 'Damping', 'Turbulence'):
        bind(cuda, nm, nm)
    # keep the rendered instance count in sync with the particle count
    bind(geo, 'numinstances', 'Count')
    # camera distance + bloom size
    bind(cam, 'tz', 'Camdist')
    bind(blur, 'size', 'Bloom')

    # GLSL uniforms: uColor (vec0), uSprite (vec1), bound to the look controls
    mat.par.vec0name = 'uColor'
    bind(mat, 'vec0valuex', 'Colorr')
    bind(mat, 'vec0valuey', 'Colorg')
    bind(mat, 'vec0valuez', 'Colorb')
    bind(mat, 'vec0valuew', 'Glow')    # per-particle intensity
    mat.par.vec1name = 'uSprite'
    bind(mat, 'vec1valuex', 'Spritesize')

    # Reset pulse -> re-seed the sim (relay to the CUDA TOP's Reset pulse)
    pex = box.create(parameterexecuteDAT, 'reset_exec')
    pex.nodeX, pex.nodeY = 0, 400
    pex.par.executeloc = 'op'
    pex.par.op = '..'
    pex.par.pars = 'Reset'
    pex.par.onpulse = True
    pex.par.valuechange = False
    pex.text = ("# Relay the component's Reset pulse to the CUDA sim.\n"
                "def onPulse(par):\n"
                "    op('cuda_particles').par.Reset.pulse()\n"
                "    return\n")


def build():
    root = op('/')
    if root.op(CONTAINER_NAME):
        root.op(CONTAINER_NAME).destroy()

    box = root.create(baseCOMP, CONTAINER_NAME)
    box.nodeX, box.nodeY = 0, 0

    # --- CUDA simulation TOP ------------------------------------------------
    cuda = box.create(cplusplusTOP, 'cuda_particles')
    cuda.nodeX, cuda.nodeY = 0, 0
    cuda.par.plugin = DLL_PATH
    try:
        cuda.cook(force=True)
    except Exception:
        pass

    # --- shaders ------------------------------------------------------------
    vdat = box.create(textDAT, 'vertex_shader')
    vdat.text = VERTEX_SHADER
    vdat.nodeX, vdat.nodeY = 0, 250

    pdat = box.create(textDAT, 'pixel_shader')
    pdat.text = PIXEL_SHADER
    pdat.nodeX, pdat.nodeY = 0, 150

    mat = box.create(glslMAT, 'particle_mat')
    mat.nodeX, mat.nodeY = 250, 200
    _set_first(mat, ['vdat', 'vertexdat'], vdat.name, 'vertex shader DAT')
    _set_first(mat, ['pdat', 'pixeldat'],  pdat.name, 'pixel shader DAT')
    _set_first(mat, ['sampler0name', 'sampler1name'], 'sPos', 'sampler name')
    _set_first(mat, ['sampler0top',  'sampler1top'],  cuda.path, 'sampler TOP')
    # Additive blending: overlapping sprites accumulate into a bright neon core
    # (the pixel shader outputs premultiplied colour, so src=one, dst=one adds
    # colour*alpha). Depth writing off so particles never occlude each other.
    _set_first(mat, ['blending'],     True,  'enable blending')
    _set_first(mat, ['blendop'],      'add', 'blend operation')
    _set_first(mat, ['srcblend'],     'one', 'source blend factor')
    _set_first(mat, ['destblend'],    'one', 'dest blend factor')
    _set_first(mat, ['depthwriting'], False, 'disable depth writing')

    # --- geometry: a unit quad, instanced N times as billboards -------------
    geo = box.create(geometryCOMP, 'cloud')
    geo.nodeX, geo.nodeY = 450, 0
    for child in list(geo.children):
        child.destroy()

    rect = geo.create(rectangleSOP, 'shape')
    _set_first(rect, ['sizex'], 1.0, 'rect size x')
    _set_first(rect, ['sizey'], 1.0, 'rect size y')
    _set_first(rect, ['orient', 'orientation'], 'xy', 'rect orientation')
    # A freshly created SOP has render/display OFF, so the geometry COMP would
    # draw nothing; enable them explicitly.
    rect.render = True
    rect.display = True

    geo.par.instancing = True
    _set_first(geo, ['instancecountmode'], 'manual', 'instance count mode')
    geo.par.numinstances = NUM_PARTICLES
    geo.par.material = mat.path

    # --- camera + light -----------------------------------------------------
    cam = box.create(cameraCOMP, 'cam1')
    cam.nodeX, cam.nodeY = 450, 200
    cam.par.tz = CAM_DIST
    cam.par.rx = 0

    light = box.create(lightCOMP, 'light1')
    light.nodeX, light.nodeY = 450, 350

    # --- render + bloom -----------------------------------------------------
    render = box.create(renderTOP, 'render1')
    render.nodeX, render.nodeY = 700, 0
    render.par.camera = cam.path
    render.par.geometry = geo.path
    render.par.lights = light.path
    _set_first(render, ['outputresolution'], 'custom', 'render resolution mode')
    _set_first(render, ['resolutionw'], 1280, 'render width')
    _set_first(render, ['resolutionh'], 720, 'render height')

    blur = box.create(blurTOP, 'bloom_blur')
    blur.nodeX, blur.nodeY = 900, -120
    blur.inputConnectors[0].connect(render)
    _set_first(blur, ['size'], BLOOM_SIZE, 'blur size')

    comp = box.create(compositeTOP, 'bloom_comp')
    comp.nodeX, comp.nodeY = 1100, 0
    comp.inputConnectors[0].connect(render)
    comp.inputConnectors[1].connect(blur)
    _set_first(comp, ['operand'], 'add', 'composite operand')

    out = box.create(nullTOP, 'out')
    out.nodeX, out.nodeY = 1300, 0
    out.inputConnectors[0].connect(comp)
    out.viewer = True

    # --- expose all controls on the COMP's own parameter page ---------------
    _expose(box, cuda, geo, mat, cam, blur)

    print(f"[cuda cloud] built /{CONTAINER_NAME} ({NUM_PARTICLES} particles). "
          f"Controls are on the COMP's 'Particle Cloud' parameter page.")
    return box


build()
