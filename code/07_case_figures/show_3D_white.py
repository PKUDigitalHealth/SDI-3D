import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.preprocessing import MinMaxScaler

# ===================================
# 解决中文显示乱码（方块）问题
# ===================================
import matplotlib
from matplotlib import font_manager as _fm

_PREFERRED_CJK = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'Noto Sans CJK SC']
_available = {f.name for f in _fm.fontManager.ttflist}
_picked = [f for f in _PREFERRED_CJK if f in _available]
# 把可用 CJK 字体放在 sans-serif 列表最前；同时保留默认 DejaVu 作为兜底
plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.sans-serif']  = _picked + ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================================
# 1. 读取数据
# ===================================
csv_path = os.environ.get(
    "SHOW_3D_CSV_PATH",
    r"C:\Users\SaberShuo\Desktop\case_study\shhs1-200419_features_predictions.csv",
)
df = pd.read_csv(csv_path)

# Support both the original PSG-evaluation CSV and the label-free real-world
# inference CSV. Real-world normalized axes are used as written, without a
# second recording-level rescaling.
legacy_columns = {'window_center', 'wake_depth_score', 'rem_prob_rem', 'label'}
real_world_columns = {
    'elapsed_seconds', 'body_score_normalized', 'rem_score_normalized'
}
if legacy_columns.issubset(df.columns):
    data = df[list(legacy_columns)].dropna().copy()
    scaler = MinMaxScaler()
    data[['wake_depth_norm', 'rem_prob_norm']] = scaler.fit_transform(
        data[['wake_depth_score', 'rem_prob_rem']]
    )
    time_sec = data['window_center'].to_numpy(dtype=float) * 30.0
    body_depth = data['wake_depth_norm'].to_numpy(dtype=float)
    brain_depth = data['rem_prob_norm'].to_numpy(dtype=float)
    stage = data['label'].to_numpy(dtype=int)
    color_mode = 'stage'
elif real_world_columns.issubset(df.columns):
    data = df[list(real_world_columns)].dropna().copy()
    time_sec = data['elapsed_seconds'].to_numpy(dtype=float)
    body_depth = data['body_score_normalized'].to_numpy(dtype=float)
    brain_depth = data['rem_score_normalized'].to_numpy(dtype=float)
    stage = None
    color_mode = 'quadrant'
else:
    raise ValueError(
        'Unsupported CSV schema. Expected either the PSG columns '
        f'{sorted(legacy_columns)} or real-world columns {sorted(real_world_columns)}.'
    )

# 转换为相对时间（从 0 开始），使三轴在左下角共享同一个 0 点
time_rel = time_sec - time_sec.min()

# ===================================
# 3. 统计计算 (穿过平面、线条长度、四个象限比例)
# ===================================
# a) 四个象限点的比例 (基于 0.5 划分)
q1 = np.mean((body_depth >= 0.5) & (brain_depth >= 0.5))
q2 = np.mean((body_depth <  0.5) & (brain_depth >= 0.5))
q3 = np.mean((body_depth <  0.5) & (brain_depth <  0.5))
q4 = np.mean((body_depth >= 0.5) & (brain_depth <  0.5))

# b) 穿过十字平面的次数
cross_body  = np.sum((body_depth[:-1]  - 0.5) * (body_depth[1:]  - 0.5) < 0)
cross_brain = np.sum((brain_depth[:-1] - 0.5) * (brain_depth[1:] - 0.5) < 0)
total_crossings = cross_body + cross_brain

# c) 三维空间线条的总长度（归一化后）
time_norm = time_rel / time_rel.max()
dx = np.diff(time_norm)
dy = np.diff(body_depth)
dz = np.diff(brain_depth)
total_length_3d = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

# d) 按小时均值
duration_hours    = time_rel.max() / 3600.0
crossings_per_hour = total_crossings  / duration_hours
length_per_hour    = total_length_3d  / duration_hours

# ===================================
# 4. 象限颜色定义
# ===================================
quadrant_meta = {
    'Q1': ('#d62728', 'Q1 (Body+, Brain+)'),   # 红
    'Q2': ('#169c9c', 'Q2 (Body−, Brain+)'),    # 青
    'Q3': ('#55a868', 'Q3 (Body−, Brain−)'),    # 绿
    'Q4': ('#d99a00', 'Q4 (Body+, Brain−)'),    # 黄
}

def get_quadrant_color(bd, brd):
    if   bd >= 0.5 and brd >= 0.5: return quadrant_meta['Q1'][0]
    elif bd <  0.5 and brd >= 0.5: return quadrant_meta['Q2'][0]
    elif bd <  0.5 and brd <  0.5: return quadrant_meta['Q3'][0]
    else:                           return quadrant_meta['Q4'][0]

scatter_colors = [get_quadrant_color(bd, brd) for bd, brd in zip(body_depth, brain_depth)]

# ===================================
# 5. 绘图风格与轴比例设置
# ===================================
FIG_SIZE = (14.4, 6.2)   # Wider, shorter canvas for a compact paper figure.
FIG_DPI  = 120           # 屏幕显示 dpi；保存仍用 SAVE_DPI
SAVE_DPI = 800

# 图内元素统一缩放（改 UI_SCALE 即可全局放大/缩小字体、线宽、点大小）
UI_SCALE = 1.65  # Main plot scale.
FS_TICK      = 9    * UI_SCALE
FS_TICK_X    = 8.0  * UI_SCALE
FS_AXIS_NUM  = 8.5  * UI_SCALE
RIGHT_FONT_SCALE = 1.40   # Right-column text must survive reduction in a paper.
FS_LEGEND    = 7    * UI_SCALE * RIGHT_FONT_SCALE
FS_STATS     = 7    * UI_SCALE * RIGHT_FONT_SCALE
FS_KEY_LABEL = 8    * UI_SCALE * RIGHT_FONT_SCALE
FS_KEY_TITLE = 9    * UI_SCALE * RIGHT_FONT_SCALE
LW_TRAJECTORY  = 1.0  * UI_SCALE
LW_LEGEND_LINE = 2.0  * UI_SCALE
LW_CENTER_AXIS = 1.5  * UI_SCALE
LW_CROSS_LINE  = 1.0  * UI_SCALE
LW_KEY_AXIS    = 2.0  * UI_SCALE
MS_SCATTER     = 6    * UI_SCALE
MS_START_END   = 100  * UI_SCALE
MS_LEGEND_DOT  = 6    * UI_SCALE

# 主图 3D 盒比例：aspect 控制三轴相对长短；zoom 过大时 artist 会被过早裁切（mpl #23450）
AXIS_BOX_ASPECT = (4.5, 1.0, 0.95)
AXIS_BOX_ZOOM   = 1.43   # Leave room for the x=0 projection and manual Y/Z tick labels.
# Y/Z 手写刻度在负坐标，需扩展 limits，否则贴左缘像被「截掉」
AXIS_LIM_PAD    = 0.14
SAVE_PAD_INCHES = 0.04   # Preserve the left projection/tick labels in exported figures.


def _disable_3d_clipping(ax3d):
    """关闭 3D artist 的 clipbox，避免 zoom/box_aspect 导致数据在盒内被硬裁切。"""
    for artist in (*ax3d.collections, *ax3d.lines, *ax3d.patches, *ax3d.texts):
        artist.set_clip_on(False)

plt.style.use('seaborn-v0_8-whitegrid')
# 画布物理裁窄；左右各 ~2% 小留白避免内容顶到边
fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
ax  = fig.add_subplot(111, projection='3d')

ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# 主图变宽 → 重新匹配 rect 横纵比；zoom 拉长三轴显示，不改变 set_*lim 与刻度
try:
    ax.set_box_aspect(AXIS_BOX_ASPECT, zoom=AXIS_BOX_ZOOM)
except TypeError:
    ax.set_box_aspect(AXIS_BOX_ASPECT)
ax.view_init(elev=20, azim=-50)
# 正交投影 + 关闭交互旋转，固定视角
try:
    ax.set_proj_type('ortho')
except Exception:
    pass
try:
    ax.disable_mouse_rotation()
except Exception:
    pass

# ===================================
# 6. 绘制 0.5 十字相交平面
# ===================================
T_MAX = time_rel.max()

# Body=0.5 平面（X-Z 面）
xx, zz = np.meshgrid([0, T_MAX], [0, 1])
yy = np.full_like(xx, 0.5)
ax.plot_surface(xx, yy, zz, color='cyan', alpha=0.1, zorder=1)

# Brain=0.5 平面（X-Y 面）
xx2, yy2 = np.meshgrid([0, T_MAX], [0, 1])
zz2 = np.full_like(xx2, 0.5)
ax.plot_surface(xx2, yy2, zz2, color='magenta', alpha=0.1, zorder=1)

# 中心轴（两平面交线）
ax.plot([0, T_MAX], [0.5, 0.5], [0.5, 0.5],
        color='#404040', linewidth=LW_CENTER_AXIS, linestyle='-.', zorder=2, alpha=0.75,
        label='Center Axis (0.5)')

# ===================================
# 7. 左侧面（x=0）象限投影 + 0.5 分界十字线
# ===================================
# 背景浅色左侧面，方便观察投影
xx_l = np.array([[0, 0], [0, 0]])
yy_l = np.array([[0, 1], [0, 1]])
zz_l = np.array([[0, 0], [1, 1]])
ax.plot_surface(xx_l, yy_l, zz_l, color='#d9d9d9', alpha=0.20, zorder=0)

# 象限分界线（左侧面 0.5 十字）
ax.plot([0, 0], [0.5, 0.5], [0, 1], color='#555555', linewidth=LW_CROSS_LINE, linestyle='--', alpha=0.70, zorder=2)
ax.plot([0, 0], [0, 1], [0.5, 0.5], color='#555555', linewidth=LW_CROSS_LINE, linestyle='--', alpha=0.70, zorder=2)

# 象限散点投影到左侧面
ax.scatter([0]*len(body_depth), body_depth, brain_depth,
           c=scatter_colors, s=MS_SCATTER, alpha=0.55, zorder=3, depthshade=False)

# ===================================
# 8. 连续多色轨迹线绘制
# ===================================
stage_names = {
    0: "Wake",
    1: "N1 (Light)",
    2: "N2 (Light)",
    3: "N3 (Deep)",
    4: "REM"
}

stage_colors = {
    0: "#ff9800",   # 橙黄 Wake
    1: "#008c9e",   # 青蓝 N1，白底可见
    2: "#1976d2",   # 蓝 N2，白底可见
    3: "#9c27b0",   # 紫色 N3
    4: "#4caf50"    # 翠绿 REM
}

points   = np.array([time_rel, body_depth, brain_depth]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
if color_mode == 'stage':
    segment_colors = [stage_colors.get(s, "#777777") for s in stage[:-1]]
else:
    segment_colors = [
        get_quadrant_color(bd, brd)
        for bd, brd in zip(body_depth[:-1], brain_depth[:-1])
    ]

lc = Line3DCollection(segments, colors=segment_colors, linewidths=LW_TRAJECTORY, alpha=0.9, zorder=4)
ax.add_collection3d(lc)

if color_mode == 'stage':
    for stg in sorted(np.unique(stage)):
        if stg in stage_colors:
            ax.plot([], [], [], color=stage_colors[stg], linewidth=LW_LEGEND_LINE, label=stage_names[stg])

# ===================================
# 9. 坐标轴设置：图内不写轴名（改到右侧「坐标轴说明」面板），仅保留刻度
# ===================================
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

# 标题已按需求移除

ax.set_xlim(0, T_MAX)
ax.set_ylim(-AXIS_LIM_PAD, 1.0)
ax.set_zlim(-AXIS_LIM_PAD, 1.0)

# 时间轴刻度略小；Y/Z 刻度为下方手写 text，单独设字号
ax.tick_params(colors="#333333", labelsize=FS_TICK)
ax.tick_params(axis="x", colors="#333333", labelsize=FS_TICK_X, pad=3)

# ------------------------------------------------------------------
# Y / Z 轴刻度数字移到左侧面（x=0 面）手动绘制
# 隐藏原来右侧的刻度数字
# ------------------------------------------------------------------
ax.set_yticklabels([])
ax.set_zticklabels([])

tick_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
nonzero_ticks = [tv for tv in tick_vals if tv != 0.0]

# Y 轴刻度：沿左侧面底边（跳过 0.0，避免与 Z 的 0.0 重叠）
for tv in nonzero_ticks:
    ax.text(0, tv, -0.10,
            f'{tv:.1f}',
            color='#333333', fontsize=FS_AXIS_NUM, ha='center', va='top',
            zorder=11)

# Z 轴刻度：沿左侧面左边（跳过 0.0）
for tv in nonzero_ticks:
    ax.text(0, -0.10, tv,
            f'{tv:.1f}',
            color='#333333', fontsize=FS_AXIS_NUM, ha='right', va='center',
            zorder=11)

# Y / Z 共用同一个 0.0，画在原点斜下方
ax.text(0, -0.05, -0.05,
        '0.0',
        color='#333333', fontsize=FS_AXIS_NUM, ha='right', va='top',
        zorder=11)

# 网格线
ax.xaxis._axinfo["grid"]['color']     = (0, 0, 0, 0.14)
ax.yaxis._axinfo["grid"]['color']     = (0, 0, 0, 0.14)
ax.zaxis._axinfo["grid"]['color']     = (0, 0, 0, 0.14)
ax.xaxis._axinfo["grid"]['linestyle'] = '--'
ax.yaxis._axinfo["grid"]['linestyle'] = '--'
ax.zaxis._axinfo["grid"]['linestyle'] = '--'

# 面板底色
ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.28))
ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.28))
ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.28))

# ===================================
# 10. 起点 / 终点高亮
# ===================================
ax.scatter(0,          body_depth[0],  brain_depth[0],  color="#1b8f3a", s=MS_START_END, marker='*', edgecolor="black", linewidths=0.8, zorder=10, label="Start")
ax.scatter(time_rel[-1], body_depth[-1], brain_depth[-1], color="red",  s=MS_START_END, marker='X', edgecolor="black", linewidths=0.8, zorder=10, label="End")

# ===================================
# 11. 图例（睡眠阶段 + 象限颜色）
# ===================================
for qk, (qc, ql) in quadrant_meta.items():
    ax.plot([], [], 'o', color=qc, markersize=MS_LEGEND_DOT, label=ql)

# ===================================
# 12. 右侧：小型空 3D 坐标轴（无数据，仅示意三轴含义）
# ===================================
def _draw_empty_3d_axis_key(axk):
    # 背景由下层 FancyBboxPatch（card）提供，这里保持透明以便三块统一外观
    axk.set_facecolor("none")
    axk.set_xlim(0, 1)
    axk.set_ylim(0, 1)
    axk.set_zlim(0, 1)
    axk.set_box_aspect((1, 1, 1))
    axk.view_init(elev=20, azim=-50)
    try:
        axk.set_proj_type('ortho')
    except Exception:
        pass
    try:
        axk.disable_mouse_rotation()
    except Exception:
        pass
    axk.grid(False)
    axk.set_xticks([])
    axk.set_yticks([])
    axk.set_zticks([])
    for axis in (axk.xaxis, axk.yaxis, axk.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_color((1, 1, 1, 0))
    # 仅三根坐标射线，无数据
    L = 0.82
    axk.plot([0, L], [0, 0], [0, 0], color="#2878b5", lw=LW_KEY_AXIS)
    axk.plot([0, 0], [0, L], [0, 0], color="#2e8b57", lw=LW_KEY_AXIS)
    axk.plot([0, 0], [0, 0], [0, L], color="#7b4fa3", lw=LW_KEY_AXIS)
    axk.text(L + 0.02, 0, 0, "Time", color="#2878b5", fontsize=FS_KEY_LABEL, va="center")
    axk.text(0, L + 0.02, 0, "Body", color="#2e8b57", fontsize=FS_KEY_LABEL, ha="center", va="bottom")
    axk.text(0, 0, L + 0.02, "Brain", color="#7b4fa3", fontsize=FS_KEY_LABEL, ha="center", va="bottom")


stats_text = (
    f"--- Stats (/h) ---\n"
    f"Duration: {duration_hours:.2f} h\n"
    f"Crossings: {crossings_per_hour:.1f} /h\n"
    f"3D Length: {length_per_hour:.2f} /h\n\n"
    f"--- Quadrants ---\n"
    f"Q1 (Body+, Brain+): {q1*100:.1f}%\n"
    f"Q2 (Body-, Brain+): {q2*100:.1f}%\n"
    f"Q3 (Body-, Brain-): {q3*100:.1f}%\n"
    f"Q4 (Body+, Brain-): {q4*100:.1f}%"
)

# ===================================
# 13. 整体布局
#   左：主 3D 图（rect 横纵比与盒投影匹配，无黑边）
#   右：三块统一外观（同色卡片 + 细边框），同一列宽对齐
# ===================================
MAIN_X0  = 0.030
MAIN_Y0  = -0.020
MAIN_W   = 0.625
MAIN_H   = 1.020
RIGHT_X0 = MAIN_X0 + MAIN_W              # 0.765
RIGHT_X1 = 0.992                         # 右侧留白
RIGHT_GAP = 0.002                        # 主图与右栏间距

plt.subplots_adjust(left=MAIN_X0, right=RIGHT_X0, top=0.990, bottom=0.010)
ax.set_position([MAIN_X0, MAIN_Y0, MAIN_W, MAIN_H])

# 右栏内容直接挂在画布上（无卡片背景 / 边框）
COL_X0 = RIGHT_X0 + RIGHT_GAP
COL_X1 = RIGHT_X1
COL_XC = (COL_X0 + COL_X1) / 2

LEG_Y0,  LEG_Y1   = 0.745, 0.985
KEY_Y0,  KEY_Y1   = 0.410, 0.675
STAT_Y0, STAT_Y1  = 0.050, 0.390

# ---- 图例（无 frame，直接画布上）----
legend = ax.legend(
    loc="center",
    bbox_to_anchor=(COL_XC, (LEG_Y0 + LEG_Y1) / 2),
    bbox_transform=fig.transFigure,
    ncol=2,
    columnspacing=0.60,
    handletextpad=0.40,
    borderaxespad=0.20,
    labelspacing=0.28,
    frameon=False,
    fontsize=FS_LEGEND,
    markerscale=1.15,
)
plt.setp(legend.get_texts(), color="#222222")

# ---- Axes Key：透明背景的 3D 轴 ----
ax_axis_key = fig.add_axes(
    [COL_X0, KEY_Y0, COL_X1 - COL_X0, KEY_Y1 - KEY_Y0],
    projection="3d",
)
_draw_empty_3d_axis_key(ax_axis_key)

# Draw the title outside the 3D axes so it cannot overlap the REM label.
fig.text(
    COL_XC, 0.720,
    "Axes Key",
    transform=fig.transFigure,
    ha="center", va="top",
    color="#222222", fontsize=FS_KEY_TITLE, fontweight="bold",
)

# 布局定稿后再关裁切，减轻左侧面/轨迹在盒缘被截断
_disable_3d_clipping(ax)

# ---- 统计（无 bbox，居中文字）----
fig.text(
    COL_XC, (STAT_Y0 + STAT_Y1) / 2,
    stats_text,
    transform=fig.transFigure,
    color="#222222", fontsize=FS_STATS,
    ha="center", va="center",
    family="monospace",
    linespacing=1.00,
)

# ===================================
# 13. 保存到本地（与 CSV 同目录，*_3D.png + *_3D.pdf）
# ===================================
_out_dir = os.environ.get(
    "SHOW_3D_OUTPUT_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)
os.makedirs(_out_dir, exist_ok=True)
_out_base = os.path.join(
    _out_dir,
    os.path.splitext(os.path.basename(csv_path))[0] + "_3D_white",
)
_save_kw = dict(
    facecolor=fig.get_facecolor(),
    edgecolor="none",
    pad_inches=SAVE_PAD_INCHES,
)
_out_png = _out_base + ".png"
_out_pdf = _out_base + ".pdf"
fig.savefig(_out_png, dpi=SAVE_DPI, **_save_kw)
fig.savefig(_out_pdf, **_save_kw)
print(f"[saved] {_out_png}")
print(f"[saved] {_out_pdf}")

if os.environ.get("SHOW_3D_NO_SHOW") == "1":
    plt.close(fig)
else:
    plt.show()
