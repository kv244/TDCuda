"""
build_particle_cloud_cuda.py  --  instanced sprites (quad|triangle), stateful sim, exposed UI
=======================================================================
Builds the CUDA particle cloud in TouchDesigner as a self-contained,
GUI-controllable component.

ParticleCloudTOP.dll (CUDA, stateful) integrates position+velocity every
frame and outputs an RGBA32F position texture. A GLSL MAT reads that texture
in the vertex shader (texelFetch by TDInstanceID) and draws each particle as
a camera-facing sprite, in one of two switchable render modes:

  * Billboard quad (default, robust): each particle is a 1x1 quad -> 4 verts,
    2 tris per particle. The round sprite is carved out in the fragment stage.
  * Triangle sprite (lighter geometry, for 1M+): each particle is a single
    triangle -> 3 verts, 1 tri. ~25% fewer vertices and 50% fewer triangles
    than the quad, which is what buys headroom toward 1M+ on Ada (RTX 4070,
    sm_89). The triangle circumscribes the sprite disc; the fragment shader
    discards outside the inscribed circle, so it still looks like a round dot.

WHY NOT TRUE GL_POINTS? TouchDesigner's Render TOP rasterizes *primitives*
(polys/lines) -- it does not draw loose points, and there is no gl_PointSize
path through the MAT/Render TOP pipeline (verified: a points-only SOP, with or
without instancing, renders nothing). So the cheapest *renderable* 1-primitive
sprite TD supports is a single triangle; that is the practical substitute for
GL point sprites here. The CUDA side is identical in both modes.

Both modes read the SAME CUDA texture; only the instanced primitive + GLSL
differ. A 'Render Mode' menu on the component flips between them live (each geo
renders only in its own mode).

The /particle_cloud_cuda COMP gets its own Custom Parameter page ("Particle
Cloud"), so every control is adjustable from the component's parameter dialog
at runtime.

RUN: paste into a Text DAT -> right-click -> Run Script. View the 'out' TOP.
Re-running rebuilds /particle_cloud_cuda cleanly.
"""

# ----------------------------------------------------------------------
# CONFIG  (these become the DEFAULTS of the component's custom parameters)
# ----------------------------------------------------------------------
DLL_PATH      = r"C:/Users/julia/Downloads/TDClaude/build/Release/ParticleCloudTOP.dll"

NUM_PARTICLES = 200000         # both modes are cheap; the triangle mode pushes
                               #   comfortably past 1,000,000 on the 4070
CLOUD_RADIUS  = 6.0
SIM_SPEED     = 1.0
GRAVITY       = 0.8
SWIRL         = 1.4
DAMPING       = 0.985
TURBULENCE    = 0.4

SPRITE_SIZE   = 0.10           # world-space size of each sprite (both modes)
GLOW          = 0.06           # per-particle intensity (additive accumulates it)
CAM_DIST      = 38.0           # pulled back so the cloud reads as a glowing orb
NEON_RGB      = (0.0, 0.92, 1.0)
BLOOM_SIZE    = 10.0

RENDER_MODE   = 'billboard'    # startup mode: 'billboard' (quad) | 'triangle'

CONTAINER_NAME = "particle_cloud_cuda"

# ----------------------------------------------------------------------
# SHADERS -- billboard quad path
#   color + sprite size come from GLSL uniforms so they are controllable at
#   runtime; uColor.rgb / uColor.a (glow) and uSprite.x are fed by the GLSL
#   MAT's vector uniforms, bound to the component params.
# ----------------------------------------------------------------------
VERTEX_SHADER = """
uniform sampler2D sPos;
uniform vec4 uSprite;      // .x = world-space sprite size, .y = color mode (0 single, 1 multi)
uniform vec4 uColor;       // .rgb = single color, .a = glow
out vec2 vUV;
out vec3 vCol;             // per-particle color handed to the fragment stage

// integer hash -> [0,1), and a cheap full-sat/val hue ramp
float hash11(uint x){
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return float(x & 0x00FFFFFFu) / 16777216.0;
}
vec3 hue2rgb(float h){
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(h + k) * 6.0 - 3.0);
    return clamp(p - 1.0, 0.0, 1.0);
}

void main()
{
    int id  = TDInstanceID();
    int w   = textureSize(sPos, 0).x;
    ivec2 t = ivec2(id % w, id / w);
    vec3 particlePos = texelFetch(sPos, t, 0).xyz;

    // single color from uColor, or a per-particle random hue when multi
    vCol = (uSprite.y > 0.5) ? hue2rgb(hash11(uint(id))) : uColor.rgb;

    vUV = P.xy + 0.5;                  // 0..1 across the sprite quad
    vec3 local = P * uSprite.x;        // scale local quad to world size
    vec4 world = TDDeform(vec4(local + particlePos, 1.0));
    gl_Position = TDWorldToProj(world);
}
"""

PIXEL_SHADER = """
uniform vec4 uColor;       // .a = glow intensity (rgb handled per-particle via vCol)
out vec4 fragColor;
in vec2 vUV;
in vec3 vCol;

void main()
{
    vec2 d = vUV * 2.0 - 1.0;          // round, soft-edged sprite
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float a = smoothstep(1.0, 0.0, r2) * uColor.a;   // dim per particle so
    fragColor = TDOutputSwizzle(vec4(vCol * a, a));       // additive accumulates
}
"""

# ----------------------------------------------------------------------
# SHADERS -- triangle sprite path
#   Instanced single triangle (verts on a unit circle, so corners sit at
#   radius 1). vD = P.xy * 2 puts the triangle's inscribed circle (incircle
#   radius 0.5) at |vD| = 1, so the fragment discard carves a round dot of the
#   same world size as the quad mode (sprite radius = 0.5 * uSprite.x).
# ----------------------------------------------------------------------
TRI_VERTEX_SHADER = """
uniform sampler2D sPos;
uniform vec4 uSprite;      // .x = world-space sprite size, .y = color mode (0 single, 1 multi)
uniform vec4 uColor;       // .rgb = single color, .a = glow
out vec2 vD;
out vec3 vCol;

float hash11(uint x){
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return float(x & 0x00FFFFFFu) / 16777216.0;
}
vec3 hue2rgb(float h){
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(h + k) * 6.0 - 3.0);
    return clamp(p - 1.0, 0.0, 1.0);
}

void main()
{
    int id  = TDInstanceID();
    int w   = textureSize(sPos, 0).x;
    ivec2 t = ivec2(id % w, id / w);
    vec3 particlePos = texelFetch(sPos, t, 0).xyz;

    vCol = (uSprite.y > 0.5) ? hue2rgb(hash11(uint(id))) : uColor.rgb;

    vD = P.xy * 2.0;                   // incircle edge at |vD| = 1
    vec3 local = P * uSprite.x;        // triangle verts on radius-(uSprite.x) circle
    vec4 world = TDDeform(vec4(local + particlePos, 1.0));
    gl_Position = TDWorldToProj(world);
}
"""

TRI_PIXEL_SHADER = """
uniform vec4 uColor;       // .a = glow intensity (rgb handled per-particle via vCol)
out vec4 fragColor;
in vec2 vD;
in vec3 vCol;

void main()
{
    float r2 = dot(vD, vD);            // inscribed-disc carve from the triangle
    if (r2 > 1.0) discard;
    float a = smoothstep(1.0, 0.0, r2) * uColor.a;
    fragColor = TDOutputSwizzle(vec4(vCol * a, a));       // premultiplied, additive
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


def _make_glsl_mat(box, name, vtext, ptext, cuda, x, y):
    """Create a GLSL MAT with additive blending that samples the CUDA TOP."""
    vdat = box.create(textDAT, name + '_vertex')
    vdat.text = vtext
    vdat.nodeX, vdat.nodeY = x, y + 60

    pdat = box.create(textDAT, name + '_pixel')
    pdat.text = ptext
    pdat.nodeX, pdat.nodeY = x, y - 60

    mat = box.create(glslMAT, name)
    mat.nodeX, mat.nodeY = x + 250, y
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
    return mat


def _expose(box, cuda, geo, geo_tri, mat, mat_tri, cam, blur):
    """Add a Custom Parameter page to the COMP and wire every control to it."""
    pg = box.appendCustomPage('Particle Cloud')

    # --- render mode (live switch between the two geometry paths) ---
    rm = pg.appendMenu('Rendermode', label='Render Mode')
    rm[0].menuNames  = ['billboard', 'triangle']
    rm[0].menuLabels = ['Billboard quad (robust)', 'Triangle sprite (1M+)']
    rm[0].val = RENDER_MODE

    # --- simulation controls (bind -> CUDA TOP) ---
    pg.appendInt('Count',      label='Particle Count')[0].val = NUM_PARTICLES
    pg.appendFloat('Radius',   label='Cloud Radius')[0].val   = CLOUD_RADIUS
    pg.appendFloat('Speed')[0].val                            = SIM_SPEED
    pg.appendFloat('Gravity',  label='Gravity (+pull / -push)')[0].val = GRAVITY
    pg.appendFloat('Swirl')[0].val                            = SWIRL
    pg.appendFloat('Damping')[0].val                          = DAMPING
    pg.appendFloat('Turbulence')[0].val                       = TURBULENCE
    pg.appendPulse('Reset')
    pg.appendToggle('Autoreset', label='Auto Reset')
    pg.appendFloat('Resetperiod', label='Reset Period (s)')[0].val = 4.0

    # --- look controls (shared by both render modes) ---
    cm = pg.appendMenu('Colormode', label='Color Mode')
    cm[0].menuNames  = ['single', 'multi']
    cm[0].menuLabels = ['Single color', 'Many colors']
    cm[0].val = 'single'
    col = pg.appendRGB('Color')
    col[0].val, col[1].val, col[2].val = NEON_RGB
    pg.appendFloat('Spritesize', label='Sprite Size')[0].val    = SPRITE_SIZE
    pg.appendFloat('Glow',       label='Glow Intensity')[0].val = GLOW
    pg.appendFloat('Camdist',    label='Camera Distance')[0].val = CAM_DIST
    pg.appendFloat('Bloom')[0].val                               = BLOOM_SIZE

    # nice slider ranges
    ranges = {'Count': (1024, 2097152), 'Radius': (0.1, 50), 'Speed': (0, 5),
              'Gravity': (-5, 5), 'Swirl': (0, 8), 'Damping': (0.9, 1.0),
              'Turbulence': (0, 5), 'Resetperiod': (0.5, 30),
              'Spritesize': (0.01, 1.0), 'Glow': (0, 1),
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
    # keep both geometry paths' instance count in sync with the particle count
    bind(geo, 'numinstances', 'Count')
    bind(geo_tri, 'numinstances', 'Count')
    # camera distance + bloom size
    bind(cam, 'tz', 'Camdist')
    bind(blur, 'size', 'Bloom')

    # both materials share the same uniforms: uColor (vec0), uSprite (vec1)
    for m in (mat, mat_tri):
        m.par.vec0name = 'uColor'
        bind(m, 'vec0valuex', 'Colorr')
        bind(m, 'vec0valuey', 'Colorg')
        bind(m, 'vec0valuez', 'Colorb')
        bind(m, 'vec0valuew', 'Glow')    # per-particle intensity
        m.par.vec1name = 'uSprite'
        bind(m, 'vec1valuex', 'Spritesize')
        # uSprite.y carries the color mode (0 = single, 1 = multi). This is a
        # computed value, not a parameter reference, so it uses EXPRESSION mode
        # (BIND expects a Par object; a bare int is not bindable).
        m.par.vec1valuey.mode = ParMode.EXPRESSION
        m.par.vec1valuey.expr = "1 if parent().par.Colormode.eval() == 'multi' else 0"

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

    # Auto-reset: re-pulse Reset every 'Reset Period' seconds while Auto Reset
    # is on, so a dispersing cloud (e.g. negative gravity) explodes and reforms
    # on a loop. absTime-based, so it tracks wall-clock rather than frame count.
    box.store('lastreset', absTime.seconds)
    aex = box.create(executeDAT, 'autoreset_exec')
    aex.nodeX, aex.nodeY = 250, 400
    _set_first(aex, ['framestart', 'frame'], True, 'enable onFrameStart')
    aex.text = (
        "def onFrameStart(frame):\n"
        "    b = op('..')\n"
        "    try:\n"
        "        if not b.par.Autoreset.eval():\n"
        "            return\n"
        "        period = max(0.1, b.par.Resetperiod.eval())\n"
        "        if absTime.seconds - b.fetch('lastreset', 0.0) >= period:\n"
        "            b.par.Reset.pulse()\n"
        "            b.store('lastreset', absTime.seconds)\n"
        "    except Exception:\n"
        "        pass\n"
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

    # --- materials: one per render path, both sampling the CUDA texture -----
    mat     = _make_glsl_mat(box, 'particle_mat', VERTEX_SHADER, PIXEL_SHADER,
                             cuda, 250, 200)
    mat_tri = _make_glsl_mat(box, 'tri_mat', TRI_VERTEX_SHADER,
                             TRI_PIXEL_SHADER, cuda, 250, -120)

    # --- geometry A: a unit quad, instanced N times as billboards -----------
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

    # --- geometry B: a single triangle, instanced N times -------------------
    # A circle SOP with 3 divisions is one triangular polygon (3 verts, 1 tri)
    # -- the cheapest renderable sprite TD will rasterize. ~25% fewer verts and
    # 50% fewer tris than the quad, so it carries higher particle counts before
    # becoming vertex/primitive-bound. The round dot is carved in the fragment
    # shader (TD's Render TOP won't draw loose GL_POINTS, so this is how we get
    # a point-sprite look at 1M+).
    geo_tri = box.create(geometryCOMP, 'cloud_tri')
    geo_tri.nodeX, geo_tri.nodeY = 450, -200
    for child in list(geo_tri.children):
        child.destroy()

    tri = geo_tri.create(circleSOP, 'shape')
    _set_first(tri, ['type'], 'poly', 'circle type=poly')
    _set_first(tri, ['arc'], 'full', 'circle arc=full')
    _set_first(tri, ['divs', 'divisions'], 3, 'circle divisions=3')
    _set_first(tri, ['radx'], 1.0, 'circle radius x')
    _set_first(tri, ['rady'], 1.0, 'circle radius y')
    _set_first(tri, ['texture'], False, 'circle texture coords off')
    tri.render = True
    tri.display = True

    geo_tri.par.instancing = True
    _set_first(geo_tri, ['instancecountmode'], 'manual', 'instance count mode')
    geo_tri.par.numinstances = NUM_PARTICLES
    geo_tri.par.material = mat_tri.path

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
    # both geometry paths are assigned; the render flag (bound below) decides
    # which one actually draws, so there is no double cost.
    render.par.geometry = geo.path + ' ' + geo_tri.path
    render.par.lights = light.path
    # live Billboard|Triangle switch: each geo renders only in its own mode
    geo.par.render.expr     = "parent().par.Rendermode.eval() == 'billboard'"
    geo_tri.par.render.expr = "parent().par.Rendermode.eval() == 'triangle'"
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
    _expose(box, cuda, geo, geo_tri, mat, mat_tri, cam, blur)

    print(f"[cuda cloud] built /{CONTAINER_NAME} ({NUM_PARTICLES} particles, "
          f"mode={RENDER_MODE}). Controls on the COMP's 'Particle Cloud' page; "
          f"flip 'Render Mode' to switch Billboard<->Triangle live.")
    return box


build()
