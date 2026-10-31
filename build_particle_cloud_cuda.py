"""
build_particle_cloud_cuda.py  --  billboard-sprite + stateful-sim version
=======================================================================
Builds the CUDA particle cloud in TouchDesigner.

ParticleCloudTOP.dll (CUDA, stateful) integrates position+velocity every
frame and outputs an RGBA32F position texture. A GLSL MAT reads that texture
in the vertex shader (texelFetch by TDInstanceID) and draws each particle as
a small camera-facing QUAD (billboard) sized in the vertex shader. Two
triangles per particle is far lighter than instancing sphere geometry, so
counts scale into the hundreds of thousands on a 4070, while rendering
reliably (no reliance on GL program-point-size).

RUN: paste into a Text DAT -> right-click -> Run Script. View the 'out' TOP.
Re-running rebuilds /particle_cloud_cuda cleanly.
"""

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DLL_PATH      = r"C:/Users/julia/Downloads/TDClaude/build/Release/ParticleCloudTOP.dll"

NUM_PARTICLES = 200000         # billboards are cheap -> push this high. Keep == TOP's Count.
CLOUD_RADIUS  = 6.0
SIM_SPEED     = 1.0
GRAVITY       = 0.8            # central well strength
SWIRL         = 1.4           # tangential spin
DAMPING       = 0.985         # velocity retention / frame
TURBULENCE    = 0.4

SPRITE_SIZE   = 0.13          # world-space size of each square sprite
CAM_DIST      = 22.0          # pulled back so the cloud reads as a volume
NEON_RGB      = (0.0, 0.92, 1.0)
BLOOM_SIZE    = 10.0

CONTAINER_NAME = "particle_cloud_cuda"

# ----------------------------------------------------------------------
# SHADERS  (tokens __SPRITE__ / __R__ __G__ __B__ are replaced, to avoid
#           GLSL-brace collisions with str.format)
#
# The quad's local vertex P spans [-0.5, 0.5] (Rectangle SOP, size 1). We
# scale it to SPRITE_SIZE and translate it to the particle's world position
# read from the CUDA texture. Because we only TRANSLATE the quad (never rotate
# it), it stays in the XY plane and faces the +Z-looking camera -- a cheap,
# robust billboard for this fixed front view.
# ----------------------------------------------------------------------
VERTEX_SHADER = """
uniform sampler2D sPos;
out vec2 vUV;

void main()
{
    int id  = TDInstanceID();
    int w   = textureSize(sPos, 0).x;
    ivec2 t = ivec2(id % w, id / w);
    vec3 particlePos = texelFetch(sPos, t, 0).xyz;

    vUV = P.xy + 0.5;                 // 0..1 across the sprite quad
    vec3 local = P * __SPRITE__;      // scale local quad to world sprite size
    vec4 world = TDDeform(vec4(local + particlePos, 1.0));
    gl_Position = TDWorldToProj(world);
}
"""

PIXEL_SHADER = """
out vec4 fragColor;
in vec2 vUV;

void main()
{
    // round, soft-edged sprite from the quad's UV
    vec2 d = vUV * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float alpha = smoothstep(1.0, 0.0, r2);

    vec3 col = vec3(__R__, __G__, __B__);
    fragColor = TDOutputSwizzle(vec4(col * alpha, alpha));
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
    for name, val in (('Count', NUM_PARTICLES), ('Radius', CLOUD_RADIUS),
                      ('Speed', SIM_SPEED), ('Gravity', GRAVITY),
                      ('Swirl', SWIRL), ('Damping', DAMPING),
                      ('Turbulence', TURBULENCE)):
        if hasattr(cuda.par, name):
            setattr(cuda.par, name, val)
        else:
            print(f"[cuda cloud] note: TOP param '{name}' not present yet")

    # --- shaders ------------------------------------------------------------
    vdat = box.create(textDAT, 'vertex_shader')
    vdat.text = VERTEX_SHADER.replace('__SPRITE__', f'{SPRITE_SIZE:.4f}')
    vdat.nodeX, vdat.nodeY = 0, 250

    pdat = box.create(textDAT, 'pixel_shader')
    pdat.text = (PIXEL_SHADER.replace('__R__', f'{NEON_RGB[0]:.4f}')
                             .replace('__G__', f'{NEON_RGB[1]:.4f}')
                             .replace('__B__', f'{NEON_RGB[2]:.4f}'))
    pdat.nodeX, pdat.nodeY = 0, 150

    mat = box.create(glslMAT, 'particle_mat')
    mat.nodeX, mat.nodeY = 250, 200
    _set_first(mat, ['vdat', 'vertexdat'], vdat.name, 'vertex shader DAT')
    _set_first(mat, ['pdat', 'pixeldat'],  pdat.name, 'pixel shader DAT')
    _set_first(mat, ['sampler0name', 'sampler1name'], 'sPos', 'sampler name')
    _set_first(mat, ['sampler0top',  'sampler1top'],  cuda.path, 'sampler TOP')

    # --- geometry: a unit quad, instanced N times as billboards -------------
    geo = box.create(geometryCOMP, 'cloud')
    geo.nodeX, geo.nodeY = 450, 0
    for child in list(geo.children):
        child.destroy()

    rect = geo.create(rectangleSOP, 'shape')
    _set_first(rect, ['sizex'], 1.0, 'rect size x')
    _set_first(rect, ['sizey'], 1.0, 'rect size y')
    _set_first(rect, ['orient', 'orientation'], 'xy', 'rect orientation')
    # The geometry COMP only renders child SOPs whose render/display flags are
    # on; a freshly created SOP has them OFF, so set them explicitly.
    rect.render = True
    rect.display = True

    geo.par.instancing = True
    _set_first(geo, ['instancecountmode'], 'manual', 'instance count mode')
    geo.par.numinstances = NUM_PARTICLES
    geo.par.material = mat.path
    # the sim provides all the motion; no whole-cloud auto-rotate

    # --- camera + light -----------------------------------------------------
    cam = box.create(cameraCOMP, 'cam1')
    cam.nodeX, cam.nodeY = 450, 200
    cam.par.tz = CAM_DIST
    cam.par.rx = -8

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

    print(f"[cuda cloud] built /{CONTAINER_NAME} ({NUM_PARTICLES} particles, "
          f"stateful sim + billboard sprites). View the 'out' TOP.")
    return box


build()
