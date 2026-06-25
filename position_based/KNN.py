import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from pyproj import Proj, Transformer
import numpy as np
import pandas as pd
import re
from math import radians, cos, atan2, sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter, defaultdict
import seaborn as sns
from imblearn.over_sampling import SMOTE

# ============================
# 1. 基站参数（替换成你的）
# ============================
BS_lat = 33.31101852725683       # 示例
BS_lon = -111.89238645275388     # 示例
BS_height = 1.5          # 米

# ============================
# 2. 读取 Excel
# ============================
excel_path = "motion_data.xlsx"   # <-- 换成你的路径
try:
    df = pd.read_excel(excel_path)
except FileNotFoundError:
    print(f"错误：文件未找到。请检查路径：{excel_path}")
    exit()

# ============================
# 3. 解析和坐标转换函数
# ============================
def parse_loc(loc_str):
    """
    将字符串 '[[lat],[lon]]' 解析为 (lat, lon)
    """
    try:
        arr = ast.literal_eval(loc_str)
        # 确保 arr 是包含两个列表的列表，并且每个子列表至少有一个元素
        if len(arr) >= 2 and len(arr[0]) >= 1 and len(arr[1]) >= 1:
            lat = float(arr[0][0])
            lon = float(arr[1][0])
            return lat, lon
        else:
            return np.nan, np.nan
    except (ValueError, SyntaxError, TypeError):
        return np.nan, np.nan


def extract_float(text):
    """从字符串中提取第一个浮点数"""
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
    return float(nums[0]) if len(nums) > 0 else np.nan

def gps_to_local(lat, lon, lat0, lon0):
    """
    将 GPS 坐标转换为局部 ENU (East-North-Up) 坐标 (x, y)
    """
    R = 6371000.0  # 地球平均半径
    dlat = radians(lat - lat0)
    dlon = radians(lon - lon0)
    x = R * dlon * cos(radians(lat0))
    y = R * dlat
    return x, y

# ============================
# 4. 数据处理
# ============================
# 应用解析函数
df[["lat", "lon"]] = df["unit2_loc"].apply(parse_loc).apply(pd.Series)

# 转换为局部坐标 (x, y)
df['x'], df['y'] = zip(*df.apply(lambda r: gps_to_local(r["lat"], r["lon"], BS_lat, BS_lon), axis=1))

# UAV 高度
df['z'] = df['unit2_height'].apply(extract_float)

# 波束 index
df['beam'] = df['unit1_beam_index']

# 移除任何因为解析失败而产生的 NaN 行
df.dropna(subset=['lat', 'lon', 'x', 'y', 'z'], inplace=True)


# ============================
# 计算方位角 & 俯仰角
# ============================

# 方位角 azimuth：[-π, π]
df["azimuth"] = np.arctan2(df["y"], df["x"])   # 弧度制

# 俯仰角 elevation
df["elevation"] = np.arctan2(df["z"], np.sqrt(df["x"]**2 + df["y"]**2))

# 转换为角度（可选）
df["az_deg"] = np.degrees(df["azimuth"])
df["el_deg"] = np.degrees(df["elevation"])



# df = df[df["seq_index"] == 2].reset_index(drop=True)

# # ============================
# # 5. 三横向子图可视化
# # ============================
# # 设置整体图形和子图，(1, 3) 表示 1 行 3 列
# fig = plt.figure(figsize=(20, 8),dpi=380) # 调整 figsize 使其横向更宽
# beam_list = sorted(df["beam"].unique())
# num_beams = len(beam_list)
#
# # 使用 seaborn 的颜色板，例如 'Spectral' 或 'tab10'
# cmap_name = 'Spectral'
# cmap = sns.color_palette(cmap_name, num_beams)
# color_map = {b: cmap[i] for i, b in enumerate(beam_list)}
#
#
# ## ----------------------------------------------------
# ## 子图 1: XY 平面图 (2D)
# ## ----------------------------------------------------
# ax1 = fig.add_subplot(1, 3, 1) # 1 行 3 列的第 1 个
# for b in beam_list:
#     sub = df[df["beam"] == b]
#     ax1.scatter(sub["x"], sub["y"],
#                 s=40,  # 增大点的大小，使空心圆更明显
#                 facecolors='none',  # <-- 设置为空心
#                 edgecolors=color_map[b],  # <-- 边框颜色
#                 linewidths=1.5,  # 边框宽度
#                 label=f"Beam {b}",
#                 alpha=0.8)
#
# # 绘制基站
# ax1.scatter(
#     0, 0,
#     s=800,
#     c="red",
#     marker="*",
#     edgecolors="black",
#     linewidths=1.2,
#     label="Base Station"
# )
#
# ax1.set_xlabel("X (m)", fontsize=12)
# ax1.set_ylabel("Y (m)", fontsize=12)
# ax1.set_title("UAV Positions: XY Plane", fontsize=14)
# ax1.grid(True, linestyle='--', alpha=0.6)
# # ax1.legend(loc="best", frameon=True)
# ax1.set_aspect('equal', adjustable='box') # 保持坐标轴比例一致
#
#
# ## ----------------------------------------------------
# ## 子图 2: 方位角 vs 俯仰角 (2D)
# ## ----------------------------------------------------
# ax2 = fig.add_subplot(1, 3, 2) # 1 行 3 列的第 2 个
#
# for b in beam_list:
#     sub = df[df["beam"] == b]
#     ax2.scatter(sub["az_deg"], sub["el_deg"],
#                 s=40,  # 增大点的大小
#                 facecolors='none',  # <-- 设置为空心
#                 edgecolors=color_map[b],  # <-- 边框颜色,
#                 alpha=0.7,
#                 label=f"Beam {b}")
#
# ax2.set_xlabel("Azimuth Angle (degrees)", fontsize=12)
# ax2.set_ylabel("Elevation Angle (degrees)", fontsize=12)
# ax2.set_title("Azimuth-Elevation Domain", fontsize=14)
# ax2.grid(True, linestyle='--', alpha=0.6)
# # ax2.legend(loc="best", frameon=True)
#
#
# ## ----------------------------------------------------
# ## 子图 3: 3D 可视化 (XYZ) - 美化版本
# ## ----------------------------------------------------
# ax3 = fig.add_subplot(1, 3, 3, projection="3d") # 1 行 3 列的第 3 个
#
# # 1. 绘制轨迹点
# # 由于 matplotilb 3D 散点图的颜色条不方便直接映射离散的 beam index，
# # 我们使用循环绘制来保持 Beam 和颜色的一致性。
# for b in beam_list:
#     sub = df[df["beam"] == b]
#     # 使用空心点 (marker='o', facecolors='none')，并用颜色映射边框
#     ax3.scatter(
#         sub["x"], sub["y"], sub["z"],
#         marker="o",
#         facecolors="none",            # 完全空心
#         edgecolors=color_map[b],      # 边框颜色表示 beam index
#         s=40,                         # 点大小
#         linewidths=1.5,
#         label=f"Beam {b}"
#     )
#
# # 2. 绘制基站点
# ax3.scatter(
#     0, 0, BS_height,
#     s=800,
#     c="red",
#     marker="*",
#     edgecolors="black",
#     linewidths=1.5,
#     label="Base Station"
# )
#
# # 3. 优化 3D 视图样式
# # 移除灰色背景（设置 pane 颜色为白色）
# ax3.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
# ax3.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
# ax3.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
#
# # 4. 调整视角以看清基站和轨迹
# # elev: 俯仰角 (Elevation)；azim: 方位角 (Azimuth)
# ax3.view_init(elev=20, azim=130) # 调整到基站和轨迹清晰可见的角度
#
# # 5. 标签和标题
# ax3.set_xlabel("X (m)", fontsize=12)
# ax3.set_ylabel("Y (m)", fontsize=12)
# ax3.set_zlabel("Z (m)", fontsize=12)
# ax3.set_title("3D UAV Trajectory (Colored by Beam Index)", fontsize=14)
#
# # 6. 图例
# # ax3.legend(loc='best')
# ax3.grid(True, linestyle='--', alpha=0.6)
#
#
# # ============================
# # 6. 最终显示
# # ============================
# plt.suptitle(f"UAV Motion Analysis (Data from: {excel_path})", fontsize=16, fontweight='bold', y=1.02)
# plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局以适应 Suptitle
# plt.show()




# # =============================
# # 5. 二维可视化
# # =============================
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111)
# beam_list = sorted(df["unit1_beam_index"].unique())
# cmap = plt.cm.get_cmap("tab20", len(beam_list))  # 分类色图（最多20种颜色）
# for i, b in enumerate(beam_list):
#     sub = df[df["unit1_beam_index"] == b]
#     ax.scatter(sub["x"], sub["y"],
#                s=18,
#                color=cmap(i),
#                label=f"Beam {b}")
# ax.scatter(
#     0, 0,           # 基站在局部坐标中的位置就是(0,0)
#     s=800,
#     c="red",
#     marker="*",
#     edgecolors="black",
#     linewidths=1.2,
#     label="Base Station"
# )
# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# # ax.set_zlabel("Z (m)")
# ax.set_title("UAV Positions colored by Beam Index (ENU)")
# # 如果 beam 数量很多，legend 会太大，你可以关掉或只显示部分
# # ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
# plt.tight_layout()
# plt.show()
#
# # ============================
# # 2D 可视化：方位角 vs 俯仰角
# # ============================
#
# beam_list = sorted(df["beam"].unique())
# cmap = plt.cm.get_cmap("tab20", len(beam_list))
#
# plt.figure(figsize=(10, 8))
#
# for i, b in enumerate(beam_list):
#     sub = df[df["beam"] == b]
#     plt.scatter(sub["az_deg"], sub["el_deg"],
#                 s=12, color=cmap(i), alpha=0.8,
#                 label=f"Beam {b}")
#
# plt.xlabel("Azimuth Angle (degrees)")
# plt.ylabel("Elevation Angle (degrees)")
# plt.title("UAV Samples in Azimuth-Elevation Domain (Colored by Beam Index)")
#
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # ============================
# # 5. 3D 可视化
# # ============================
# fig = plt.figure(figsize=(12, 9), dpi=300)
# ax = fig.add_subplot(111)
# ax = fig.add_subplot(111, projection="3d")
# # 使用 colormap 根据 beam 映射颜色
# colors = plt.cm.tab20(df["beam"] / df["beam"].max())
# p = ax.scatter(
#     df["x"], df["y"], df["z"],
#     marker="o",
#     facecolors="none",            # 完全空心
#     edgecolors=colors,            # 边框颜色表示 beam index
#     s=40,                         # 点大小
#     linewidths=1.5                # 边框宽度
# )
# # 基站点
# ax.scatter(
#     0, 0, BS_height,
#     s=800,
#     c="red",
#     marker="*",
#     edgecolors="black",
#     linewidths=1.2,
#     label="Base Station"
# )
#
# # 坐标标签
# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# ax.set_zlabel("Z (m)")
# ax.set_title("3D UAV Positions Colored by Beam Index (ENU Coordinates)")
# # 颜色条（beam index）
# cbar = fig.colorbar(p, ax=ax, shrink=0.6)
# cbar.set_label("Beam Index")
# plt.legend()
# plt.tight_layout()
# plt.show()


# ======================KNN-Position# =====================
# ==========================================
# 1. 特征与标签
# ==========================================
df = df.groupby("beam").filter(lambda x: len(x) >= 3)

X = df[["x", "y"]].values   # 只用 x,y 画决策边界更清晰
y = df["beam"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)


smote = SMOTE(k_neighbors=1, random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("原始训练集大小:", X_train.shape,
      "样本分布:", np.bincount(y_train))

print("SMOTE 后大小:", X_train_resampled.shape,
      "样本分布:", np.bincount(y_train_resampled))

# ==========================================
# 2. 特征归一化
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# X_train_scaled = X_train
# X_test_scaled = X_test


# ==========================================
# 3. 使用 GroupKFold + GridSearchCV 搜 K
# ==========================================
param_grid = {"n_neighbors": list(range(1, 21))}

# 默认 5-fold KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=KNeighborsClassifier(weights="distance"),
    param_grid=param_grid,
    cv=kf,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# 训练
grid.fit(X_train_scaled, y_train)

best_k = grid.best_params_["n_neighbors"]
print(f"Best K = {best_k}")
print(f"Best CV Accuracy = {grid.best_score_:.4f}")

# 使用最优 K 重新训练模型
knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"Training Accuracy with Best K = {test_acc:.4f}")


# ======== 使用上述函数计算 top-k ===========
def knn_topk_accuracy(knn, X_test, y_test, k_list=[1,3,5]):
    """
    返回多个 top-k accuracy
    """
    # 获取KNN的邻居（返回每个样本的K个邻居的 index）
    neigh_dist, neigh_ind = knn.kneighbors(X_test)

    # 邻居的标签
    neigh_labels = y_train[neigh_ind]   # (num_test, K_neighbors)

    results = {}
    for k in k_list:
        topk_correct = 0
        for i in range(len(y_test)):
            # 出现次数排序，获取vote最多的top-k类别
            counts = np.bincount(neigh_labels[i])
            top_k_beams = np.argsort(counts)[::-1][:k]

            if y_test[i] in top_k_beams:
                topk_correct += 1

        results[f"top{k}"] = topk_correct / len(y_test)

    return results


topk_results = knn_topk_accuracy(knn, X_test_scaled, y_test, k_list=[1,3,5])

print("\n===== Top-K Accuracy =====")
for k, acc in topk_results.items():
    print(f"{k} Accuracy = {acc:.4f}")




# ==========================================
# 4. 绘制 2D 决策边界（x-y 平面）
# ==========================================
# 构建网格（注意只用 TEST 范围避免训练数据泄露）
x_min, x_max = X_test_scaled[:, 0].min() - 0.3, X_test_scaled[:, 0].max() + 0.3
y_min, y_max = X_test_scaled[:, 1].min() - 0.3, X_test_scaled[:, 1].max() + 0.3

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

# 网格分类预测
zz = knn.predict(grid_points)
zz = zz.reshape(xx.shape)

plt.figure(figsize=(11, 9))

# 决策边界（背景颜色）
plt.contourf(xx, yy, zz, alpha=0.25, cmap="tab20")

# 测试集散点
plt.scatter(
    X_test_scaled[:, 0],
    X_test_scaled[:, 1],
    c=y_test,                # <—— 正确颜色
    cmap="tab20",
    s=25,
    edgecolors="k",
    linewidths=0.4
)

plt.title("KNN Beam Classification – Decision Boundary (2D XY)", fontsize=15)
plt.xlabel("X (scaled)")
plt.ylabel("Y (scaled)")

cbar = plt.colorbar()
cbar.set_label("Beam Index")

plt.tight_layout()
plt.show()


# ==========================================
# 5. Confusion Matrix（真实预测误差热力图）
# ==========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    cmap="Blues",
    annot=False,
    cbar=True,
    square=True
)

plt.title("Beam Prediction Confusion Matrix (Test Set)", fontsize=15)
plt.xlabel("Predicted Beam")
plt.ylabel("True Beam")

plt.tight_layout()
plt.show()







X = df[["azimuth", "elevation"]].values
y = df["beam"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train_scaled = X_train
X_test_scaled = X_test

param_grid = {"n_neighbors": list(range(1, 21))}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=KNeighborsClassifier(weights="distance"),
    param_grid=param_grid,
    cv=kf,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_scaled, y_train)

best_k = grid.best_params_["n_neighbors"]
print(f"Best K = {best_k}")
print(f"Best CV Accuracy = {grid.best_score_:.4f}")


knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy = {test_acc:.4f}")


def knn_topk_accuracy(knn, X_test, y_test, y_train, k_list=[1,3,5]):
    neigh_dist, neigh_ind = knn.kneighbors(X_test)
    neigh_labels = y_train[neigh_ind]

    results = {}
    for k in k_list:
        topk_correct = 0
        for i in range(len(y_test)):
            counts = np.bincount(neigh_labels[i])
            top_k_beams = np.argsort(counts)[::-1][:k]
            if y_test[i] in top_k_beams:
                topk_correct += 1

        results[f"top{k}"] = topk_correct / len(y_test)

    return results


topk_results = knn_topk_accuracy(knn, X_test_scaled, y_test, y_train, k_list=[1, 3, 5])

print("\n===== Top-K Accuracy =====")
for k, acc in topk_results.items():
    print(f"{k} Accuracy = {acc:.4f}")


# 网格范围
x_min, x_max = X_test_scaled[:, 0].min()-0.3, X_test_scaled[:, 0].max()+0.3
y_min, y_max = X_test_scaled[:, 1].min()-0.3, X_test_scaled[:, 1].max()+0.3

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]
zz = knn.predict(grid_points)
zz = zz.reshape(xx.shape)

plt.figure(figsize=(11, 9))
plt.contourf(xx, yy, zz, alpha=0.25, cmap="tab20")

plt.scatter(
    X_test_scaled[:, 0],
    X_test_scaled[:, 1],
    c=y_test,
    cmap="tab20",
    s=20,
    edgecolors="k",
    linewidths=0.3
)

plt.title("Decision Boundary in (Azimuth, Elevation)", fontsize=15)
plt.xlabel("Azimuth (scaled)")
plt.ylabel("Elevation (scaled)")

cbar = plt.colorbar()
cbar.set_label("Beam Index")
plt.tight_layout()
plt.show()


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", annot=False, square=True)
plt.title("Beam Prediction Confusion Matrix")
plt.xlabel("Predicted Beam")
plt.ylabel("True Beam")
plt.tight_layout()
plt.show()
