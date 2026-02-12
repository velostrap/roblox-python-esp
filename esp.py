# shit ahh code 

# esp made by me but ai made the frames good

# discord.gg/velostrap

import struct, threading, time, requests
from collections import deque
from ctypes import c_size_t, c_void_p, byref, sizeof, windll

import dearpygui.dearpygui as dpg
import numpy as np
import pymem
import pyMeow as pme
import win32api

print("[+] loading esppp.")


TTL_GEN    = 0.75    # generic pointers / strings
TTL_PLAYER = 1.2     # player list + local player ptr
TTL_TEAM   = 2.5     # team — changes only on rejoin
TTL_MAXHP  = 12.0    # max HP never changes mid-life
TTL_CHAR   = 0.35    # character ptr — short for fast respawn detection
FPS_WIN    = 90      # rolling FPS window size


R15_PARTS   = frozenset({"Head","UpperTorso","LowerTorso",
                          "LeftUpperArm","LeftLowerArm","LeftHand",
                          "RightUpperArm","RightLowerArm","RightHand",
                          "LeftUpperLeg","LeftLowerLeg","LeftFoot",
                          "RightUpperLeg","RightLowerLeg","RightFoot"})
R6_PARTS    = frozenset({"Head","Torso","Left Arm","Right Arm","Left Leg","Right Leg"})
# Minimal sets used when skeleton is OFF — fewer parts = fewer memory reads
R15_MIN     = frozenset({"Head","LowerTorso","LeftHand","RightHand","LeftFoot","RightFoot"})
R6_MIN      = R6_PARTS

R15_BONES = (
    ("Head","UpperTorso"), ("UpperTorso","LowerTorso"),
    ("UpperTorso","LeftUpperArm"), ("LeftUpperArm","LeftLowerArm"), ("LeftLowerArm","LeftHand"),
    ("UpperTorso","RightUpperArm"),("RightUpperArm","RightLowerArm"),("RightLowerArm","RightHand"),
    ("LowerTorso","LeftUpperLeg"), ("LeftUpperLeg","LeftLowerLeg"), ("LeftLowerLeg","LeftFoot"),
    ("LowerTorso","RightUpperLeg"),("RightUpperLeg","RightLowerLeg"),("RightLowerLeg","RightFoot"),
)
R6_BONES = (
    ("Head","Torso"),
    ("Torso","Left Arm"), ("Torso","Right Arm"),
    ("Torso","Left Leg"), ("Torso","Right Leg"),
)


S = {   
    "box":        True,
    "dot":        False,
    "health":     False,
    "fov":        False,
    "skeleton":   False,
    "fov_radius": 100,
    "col_box":      [255, 255, 255, 255],
    "col_dot":      [255,  60,  60, 255],
    "col_skel":     [255, 200,   0, 255],
    "col_fov":      [255, 255, 255, 200],
}


_col_lock = threading.Lock()
_pending  = {              # pre-seeded: render thread has valid colors on frame 1
    "box":  list(S["col_box"]),
    "dot":  list(S["col_dot"]),
    "skel": list(S["col_skel"]),
    "fov":  list(S["col_fov"]),
}
_RC = {}    # rendered pme color objects — owned ONLY by render thread

def _drain():
    """Called once per frame from render thread. ~50ns noop when nothing pending."""
    with _col_lock:
        if not _pending:
            return
        snap = dict(_pending)
        _pending.clear()
    # Build pme objects OUTSIDE lock — no blocking the UI thread
    for k, c in snap.items():
        _RC[k] = pme.new_color(c[0], c[1], c[2], c[3])


_HP_LUT: list = []

def _build_lut():
    _HP_LUT.clear()
    for i in range(101):
        r = int(255 * (1.0 - i * 0.01))
        g = int(255 * (i * 0.01))
        _HP_LUT.append(pme.new_color(r, g, 0, 255))


_CD  = {}   # addr   → (ptr,   ts)   deref-pointer cache
_CS  = {}   # addr   → (str,   ts)   string cache
_CN  = {}   # inst   → (str,   ts)   name cache
_CCL = {}   # inst   → (str,   ts)   classname cache
_CCH = {}   # inst   → (list,  ts)   children cache
_CCR = {}   # player → (int,   ts)   character ptr
_CH  = {}   # char   → (int,   ts)   humanoid ptr
_CMH = {}   # hum    → (float, ts)   max-HP
_CP  = {}   # part   → (int,   ts)   primitive ptr
_CDT = {}   # char   → dict          full char data
_CT  = {}   # player → (int,   ts)   team ptr
_CLP = [0, 0.0]    # [ptr, ts]  local player — list for in-place mutation
_CPL = [[], 0.0]   # [list, ts] player list  — list for in-place mutation


def _find_pid():
    for p in pymem.process.list_processes():
        try:
            if b"RobloxPlayerBeta.exe" in p.szExeFile:
                return p.th32ProcessID
        except Exception:
            pass
    return None

def _get_base(pid):
    h = windll.kernel32.OpenProcess(0x0410, False, pid)
    if not h:
        return None
    try:
        mods = (c_void_p * 1)()
        need = c_size_t()
        if windll.psapi.EnumProcessModules(h, byref(mods), sizeof(mods), byref(need)):
            return int(mods[0])
    finally:
        windll.kernel32.CloseHandle(h)
    return None

_pid = None
while not _pid:
    _pid = _find_pid()
    if not _pid:
        print("[~] Waiting for Roblox..."); time.sleep(1)

_pm   = pymem.Pymem(_pid)
_base = _get_base(_pid)
print(f"[+] Attached  base={hex(_base)}")

# pymem bind
_RLL  = _pm.read_longlong
_RF   = _pm.read_float
_RI   = _pm.read_int
_RB   = _pm.read_bytes
_RSTR = _pm.read_string


_raw = requests.get("https://offsets.ntgetwritewatch.workers.dev/offsets.json").json()
_OFF = {k: int(v, 16)
        for k, v in _raw.items()
        if isinstance(v, str) and v.startswith("0x") and len(v) > 2}

_miss = [k for k in ("Name","Children","LocalPlayer","ModelInstance",
                     "FakeDataModelPointer","FakeDataModelToDataModel",
                     "VisualEnginePointer","viewmatrix",
                     "Health","MaxHealth","Primitive","Position","Team")
         if k not in _OFF]
if _miss:
    print(f"[!] Missing offsets: {', '.join(_miss)}")


_ON   = _OFF.get("Name", 0)
_OCH  = _OFF.get("Children", 0)
_OLP  = _OFF.get("LocalPlayer", 0)
_OMI  = _OFF.get("ModelInstance", 0)
_OPR  = _OFF.get("Primitive", 0)
_OPS  = _OFF.get("Position", 0)
_OHP  = _OFF.get("Health", 0)
_OMHP = _OFF.get("MaxHealth", 0)
_OTM  = _OFF.get("Team", 0)
_OVE  = _OFF.get("VisualEnginePointer", 0)
_OVM  = _OFF.get("viewmatrix", 0)
_OFDM = _OFF.get("FakeDataModelPointer", 0)
_ODM  = _OFF.get("FakeDataModelToDataModel", 0)

_mono = time.monotonic   # module-level alias — one fewer attribute lookup per call




def _drp(addr: int) -> int:
    """Cached pointer dereference."""
    if not addr or addr > 0x7FFFFFFFFFFF:
        return 0
    e = _CD.get(addr)
    if e:
        v, t = e
        if _mono() - t < TTL_GEN:
            return v
    try:
        v = _RLL(addr)
    except Exception:
        v = 0
    _CD[addr] = (v, _mono())
    return v

def _rbx_str(addr: int) -> str:
    """Read a Roblox std::string (inline ≤15 or heap >15)."""
    if not addr:
        return ""
    e = _CS.get(addr)
    if e:
        v, t = e
        if _mono() - t < TTL_GEN:
            return v
    try:
        ln = _RI(addr + 0x10)
        if ln > 15:
            p = _drp(addr)
            v = _RSTR(p, ln) if p else ""
        else:
            v = _RSTR(addr, ln + 1)
    except Exception:
        v = ""
    _CS[addr] = (v, _mono())
    return v

def _name(inst: int) -> str:
    if not inst:
        return ""
    e = _CN.get(inst)
    if e:
        v, t = e
        if _mono() - t < TTL_GEN:
            return v
    v = _rbx_str(_drp(inst + _ON))
    _CN[inst] = (v, _mono())
    return v

def _classname(inst: int) -> str:
    if not inst:
        return ""
    e = _CCL.get(inst)
    if e:
        v, t = e
        if _mono() - t < TTL_GEN:
            return v
    try:
        p  = _RLL(inst + 0x18)
        p  = _RLL(p + 0x8)
        fl = _RLL(p + 0x18)
        if fl == 0x1F:
            p = _RLL(p)
        v = _rbx_str(p)
    except Exception:
        v = ""
    _CCL[inst] = (v, _mono())
    return v

def _children(inst: int) -> list:
    """
    Read Roblox children vector.
    Uses np.frombuffer + [0::2] stride — faster than Python list-comp for
    large child counts (mirrors C++ vector<Instance*> read pattern).
    """
    if not inst:
        return []
    now = _mono()
    e   = _CCH.get(inst)
    if e:
        v, t = e
        if now - t < TTL_GEN:
            return v
    start = _drp(inst + _OCH)
    if not start:
        _CCH[inst] = ([], now); return []
    beg = _drp(start)
    end = _drp(start + 8)
    diff = end - beg
    if beg >= end or diff > 65536 or diff % 8 != 0:
        _CCH[inst] = ([], now); return []
    try:
        
        raw = _RB(beg, diff)
        v   = np.frombuffer(raw, dtype=np.uint64)[0::2].tolist()
    except Exception:
        v = []
    _CCH[inst] = (v, now)
    return v

def _child_of_class(inst: int, cls: str) -> int:
    for ch in _children(inst):
        if _classname(ch) == cls:
            return ch
    return 0

def _local_player(players: int) -> int:
    now = _mono()
    if now - _CLP[1] < TTL_PLAYER and _CLP[0]:
        return _CLP[0]
    try:
        v = _RLL(players + _OLP)
    except Exception:
        v = 0
    _CLP[0] = v; _CLP[1] = now
    return v

def _character(player: int) -> int:
    if not player:
        return 0
    now = _mono()
    e   = _CCR.get(player)
    if e:
        v, t = e
        if now - t < TTL_CHAR:
            return v
    try:
        v = _RLL(player + _OMI)
    except Exception:
        v = 0
    _CCR[player] = (v, now)
    return v

def _humanoid(char: int) -> int:
    if not char:
        return 0
    e   = _CH.get(char)
    now = _mono()
    if e:
        v, t = e
        if now - t < TTL_GEN:
            return v
    v = _child_of_class(char, "Humanoid")
    _CH[char] = (v, now)
    return v

def _maxhp(hum: int) -> float:
    if not hum:
        return 0.0
    e   = _CMH.get(hum)
    now = _mono()
    if e:
        v, t = e
        if now - t < TTL_MAXHP:
            return v
    try:
        v = _RF(hum + _OMHP)
    except Exception:
        v = 0.0
    _CMH[hum] = (v, now)
    return v

def _team(player: int) -> int:
    if not player:
        return 0
    e   = _CT.get(player)
    now = _mono()
    if e:
        v, t = e
        if now - t < TTL_TEAM:
            return v
    try:
        v = _RLL(player + _OTM)
    except Exception:
        v = 0
    _CT[player] = (v, now)
    return v


_U3F = struct.Struct("<3f").unpack_from

def _pos(part: int):
    """Return (x,y,z) tuple or None. Primitive ptr cached separately."""
    if not part:
        return None
    now = _mono()
    e   = _CP.get(part)
    if e:
        prim, t = e
        if now - t >= TTL_GEN:
            prim = None   
    else:
        prim = None
    if prim is None:
        try:
            prim = _RLL(part + _OPR)
        except Exception:
            return None
        _CP[part] = (prim, now)
    if not prim:
        return None
    try:
        return _U3F(_RB(prim + _OPS, 12))
    except Exception:
        return None


_MAX_P = 20
_IB    = np.zeros((_MAX_P, 4), dtype=np.float32)   # input buffer
_IB[:, 3] = 1.0                                     # W=1 set once
_SX    = np.empty(_MAX_P, dtype=np.float32)         # output screen X
_SY    = np.empty(_MAX_P, dtype=np.float32)         # output screen Y
_VIS   = np.empty(_MAX_P, dtype=np.bool_)           # output visibility

def _w2s(positions, vm, hw, hh):
    n = len(positions)
    for i in range(n):
        _IB[i, 0], _IB[i, 1], _IB[i, 2] = positions[i]

    clip  = _IB[:n] @ vm.T               
    w     = clip[:, 3]
    mask  = w > 1e-4
    inv_w = np.where(mask, 1.0 / np.where(mask, w, 1.0), 0.0)
    ndx   = clip[:, 0] * inv_w
    ndy   = clip[:, 1] * inv_w
    vis   = mask & (np.abs(ndx) <= 1.05) & (np.abs(ndy) <= 1.05)

    _SX[:n]  = (ndx + 1.0) * hw
    _SY[:n]  = (1.0 - ndy) * hh
    _VIS[:n] = vis

    return [(int(_SX[i]), int(_SY[i])) if _VIS[i] else None for i in range(n)]


def _chardata(char: int) -> dict:
    if not char:
        return {}
    now = _mono()
    e   = _CDT.get(char)
    if e:
        if now - e["ts"] < TTL_GEN:
            hum = e["hum"]
            try:
                if _RF(hum + _OHP) <= 0:
                    del _CDT[char]; return {}
            except Exception:
                del _CDT[char]; return {}
            return e
    kids   = _children(char)
    if not kids:
        return {}
    knames = [_name(k) for k in kids]
    r15    = "UpperTorso" in knames
    sk     = S["skeleton"]
    pset   = (R15_PARTS if r15 else R6_PARTS) if sk else (R15_MIN if r15 else R6_MIN)
    parts  = {nm: inst for inst, nm in zip(kids, knames) if nm in pset}
    hum    = _humanoid(char)
    if not hum or not parts:
        return {}
    mhp = _maxhp(hum)
    if mhp <= 0:
        return {}
    e = {"parts": parts, "r15": r15, "hum": hum, "mhp": mhp, "ts": now}
    _CDT[char] = e
    return e

# draw bind

_DL  = pme.draw_line
_DR  = pme.draw_rectangle
_DRL = pme.draw_rectangle_lines
_DC  = pme.draw_circle
_DCL = pme.draw_circle_lines
_DT  = pme.draw_text

def _bone(x1, y1, x2, y2, col, blk):
    """
    Outlined bone line — mirrors C++ DrawBone() black-outline technique.
    4 offset lines in black, then 1 colored line on top.
    All calls use pre-bound _DL — zero attribute lookup.
    """
    _DL(x1+1, y1,   x2+1, y2,   blk)
    _DL(x1-1, y1,   x2-1, y2,   blk)
    _DL(x1,   y1+1, x2,   y2+1, blk)
    _DL(x1,   y1-1, x2,   y2-1, blk)
    _DL(x1,   y1,   x2,   y2,   col)


_FB = pme.new_color(20,  20,  20,  210)   
_FG = pme.new_color(100, 220, 100, 255)   
_FW = pme.new_color(255, 255, 255, 255)   

def _fpspanel(fps: float):
    n  = str(int(fps))
    lw = 45   # "FPS: " at 9px/char = 5 chars = 45px
    _DR(8, 8, lw + len(n) * 9 + 14, 22, _FB)
    _DT("FPS: ", 15,      12, 16, _FG)
    _DT(n,       15 + lw, 12, 16, _FW)


def Render():
    sw = win32api.GetSystemMetrics(0)
    sh = win32api.GetSystemMetrics(1)
    hw = sw * 0.5
    hh = sh * 0.5

    pme.overlay_init(title="ESP", fps=0, exitKey=0x23)
    _build_lut()   

    
    fdm  = _RLL(_base + _OFDM)
    dm   = _RLL(fdm + _ODM)
    plrs = _child_of_class(dm, "Players")
    ve   = _RLL(_base + _OVE)
    mta  = ve + _OVM   # view-matrix address

    col_blk = pme.get_color("black")

    
    dlts   = deque(maxlen=FPS_WIN)
    last_t = time.perf_counter()
    fps    = 0.0

    
    es     = S               # esp_state dict
    gc     = _character
    gcd    = _chardata
    gt     = _team
    glp    = _local_player
    gp     = _pos
    w2s    = _w2s
    bn     = _bone
    fpsp   = _fpspanel
    pc     = time.perf_counter
    rf     = _RF
    rb     = _RB
    fb     = np.frombuffer
    f32    = np.float32
    gcur   = win32api.GetCursorPos
    dr     = _drain
    i_     = int
    lut    = _HP_LUT
    plist  = _CPL
    mono   = _mono
    DRL    = _DRL
    DC     = _DC
    DCL    = _DCL
    DR     = _DR

    while pme.overlay_loop():

        
        nt = pc()
        d  = nt - last_t;  last_t = nt
        if d > 0:
            dlts.append(d)
        if dlts:
            fps = len(dlts) / sum(dlts)

        
        dr()

        
        c_blk  = col_blk
        c_box  = _RC.get("box",  col_blk)
        c_dot  = _RC.get("dot",  col_blk)
        c_skel = _RC.get("skel", col_blk)
        c_fov  = _RC.get("fov",  col_blk)

        try:
            
            vm = fb(rb(mta, 64), dtype=f32).reshape(4, 4)

            pme.begin_drawing()

            lp  = glp(plrs)
            lt  = gt(lp)
            cur = gcur()

            # fov circleeeee
            if es["fov"]:
                r = es["fov_radius"]
                DCL(cur[0], cur[1], r+1, c_blk)
                DCL(cur[0], cur[1], r-1, c_blk)
                DCL(cur[0], cur[1], r,   c_fov)

            
            now = mono()
            if now - plist[1] > TTL_PLAYER:
                plist[0] = _children(plrs)
                plist[1] = now

            
            box  = es["box"]
            dot  = es["dot"]
            hp   = es["health"]
            sk   = es["skeleton"]

            for player in plist[0]:
                if player == lp:
                    continue
                if lt and gt(player) == lt:
                    continue

                char = gc(player)
                if not char:
                    continue
                cd = gcd(char)
                if not cd:
                    continue

                # health read
                try:
                    chp = rf(cd["hum"] + _OHP)
                except Exception:
                    continue
                if chp <= 0:
                    continue

                
                parts  = cd["parts"]
                vn, vp = [], []
                for nm, inst in parts.items():
                    p3 = gp(inst)
                    if p3 is not None:
                        vn.append(nm)
                        vp.append(p3)
                nv = len(vp)
                if nv < 2:
                    continue

                
                spts = w2s(vp, vm, hw, hh)

                
                
                sc = {}
                x0 =  1e9;  x1 = -1e9
                y0 =  1e9;  y1 = -1e9
                for i in range(nv):
                    sp = spts[i]
                    if sp is not None:
                        sx, sy = sp
                        sc[vn[i]] = sp
                        if sx < x0: x0 = sx
                        if sx > x1: x1 = sx
                        if sy < y0: y0 = sy
                        if sy > y1: y1 = sy

                if x1 < 0:
                    continue

                bx = i_(x0) - 4
                by = i_(y0) - 4
                bw = i_(x1 - x0) + 8
                bh = i_(y1 - y0) + 8
                cx = bx + (bw >> 1)
                cy = by + (bh >> 1)

                if box:
                    DRL(bx-1, by-1, bw+2, bh+2, c_blk)
                    DRL(bx,   by,   bw,   bh,   c_box)

                if dot:
                    DC(cx, cy, 4, c_blk)
                    DC(cx, cy, 3, c_dot)

                if hp:
                    bxb   = bx - 5
                    ratio = chp / cd["mhp"]
                    idx   = i_(ratio * 100)
                    
                    if idx < 0:    idx = 0
                    elif idx > 100: idx = 100
                    fh = i_(bh * ratio)
                    DR(bxb-1, by-1,     5,  bh+2, c_blk)
                    DR(bxb,   by+bh-fh, 3,  fh,   lut[idx])

                if sk:
                    bns = R15_BONES if cd["r15"] else R6_BONES
                    for a, b in bns:
                        if a in sc and b in sc:
                            p1, p2 = sc[a], sc[b]
                            bn(p1[0], p1[1], p2[0], p2[1], c_skel, c_blk)

            fpsp(fps)
            pme.end_drawing()

        except Exception:
            pass   # keep overlay alive through any transient read error



def _tog(sender, app_data, user_data):
    S[user_data] = bool(app_data)

def _fovr(sender, app_data, user_data):
    S["fov_radius"] = int(app_data)


_CMAP = {"box": "box", "dot": "dot", "skeleton": "skel", "fov": "fov"}

def _col(sender, app_data, user_data):
    """DPG always gives floats 0.0-1.0. Convert then push to pending."""
    key = _CMAP.get(user_data)
    if not key:
        return
    rgba = [min(255, max(0, int(round(float(v) * 255)))) for v in app_data]
    with _col_lock:
        _pending[key] = rgba

def setup_ui():
    dpg.create_context()

    with dpg.window(label="python esp", width=280, height=340,
                    no_close=True, no_collapse=True):

        dpg.add_text("ESP")
        dpg.add_spacer(height=4)

        
        
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Box ESP",
                default_value=S["box"], callback=_tog, user_data="box")
            dpg.add_color_edit(label="##cbox",
                default_value=S["col_box"], callback=_col, user_data="box",
                no_inputs=True, alpha_bar=True, width=30)

        dpg.add_spacer(height=3)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Dot ESP",
                default_value=S["dot"], callback=_tog, user_data="dot")
            dpg.add_color_edit(label="##cdot",
                default_value=S["col_dot"], callback=_col, user_data="dot",
                no_inputs=True, alpha_bar=True, width=30)

        dpg.add_spacer(height=3)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Skeleton",
                default_value=S["skeleton"], callback=_tog, user_data="skeleton")
            dpg.add_color_edit(label="##cskel",
                default_value=S["col_skel"], callback=_col, user_data="skeleton",
                no_inputs=True, alpha_bar=True, width=30)

        dpg.add_spacer(height=3)
        dpg.add_checkbox(label="Health Bar",
            default_value=S["health"], callback=_tog, user_data="health")

        dpg.add_separator()
        dpg.add_spacer(height=4)
        dpg.add_text(" misc ")
        dpg.add_spacer(height=4)

        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="FOV Circle",
                default_value=S["fov"], callback=_tog, user_data="fov")
            dpg.add_color_edit(label="##cfov",
                default_value=S["col_fov"], callback=_col, user_data="fov",
                no_inputs=True, alpha_bar=True, width=30)

        dpg.add_spacer(height=3)
        dpg.add_slider_int(label="FOV Radius",
            default_value=S["fov_radius"],
            min_value=50, max_value=600,
            callback=_fovr, user_data="fov_radius")

    dpg.create_viewport(title="fck urself", width=300, height=370)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    threading.Thread(target=Render, daemon=True).start()
    setup_ui()
