from sklearn.covariance import MinCovDet
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

warnings.filterwarnings('ignore')
plt.style.use('default')
os.makedirs("论文图表", exist_ok=True)

def setup_chinese_font():
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei',
        'Arial Unicode MS', 'PingFang SC', 'Heiti TC'
    ]

setup_chinese_font()
file_path = r"E:\Desktop\指标文件.xlsx"

def calculate_vif(X):
    X = X.dropna()
    if X.shape[1] < 2:
        return pd.DataFrame(columns=["变量名", "VIF", "共线性判断"])
    vif_data = pd.DataFrame()
    vif_data["变量名"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data["共线性判断"] = vif_data["VIF"].apply(
        lambda x: "正常" if x < 5 else ("轻度共线" if x < 10 else "严重共线"))
    return vif_data.sort_values(by="VIF", ascending=False).round(2)

adf_results = {}

def adf_test(series, name, is_diff=False):
    series = series.dropna()
    if len(series) < 5:
        print(f"\nADF 平稳性检验: {name}")
        print("样本不足，跳过检验")
        return None

    reg_type = "c" if is_diff else "ct"
    note = "一阶差分｜仅截距" if is_diff else "原序列｜截距+趋势"
    result = adfuller(series, regression=reg_type, autolag="AIC")

    print(f"\nADF 平稳性检验: {name}")
    print(f"检验类型：{note}")
    print(f"ADF 统计量: {result[0]:.4f}")
    print(f"P值:        {result[1]:.4f}")
    print(f"1% 临界值:  {result[4]['1%']:.4f}")
    print(f"5% 临界值:  {result[4]['5%']:.4f}")
    print(f"10% 临界值: {result[4]['10%']:.4f}")

    is_stationary = result[1] < 0.05
    print("平稳" if is_stationary else "不平稳")
    adf_results[name] = is_stationary
    return result

def check_i1(level_name, diff_name):
    if level_name in adf_results and diff_name in adf_results:
        if not adf_results[level_name] and adf_results[diff_name]:
            print(f"{level_name} 是 I(1) 序列")
            return True
    print(f"{level_name} 不是标准 I(1) 序列")
    return False

def eg_test(data, y_col, x_cols):
    print("\n" + "=" * 80)
    print("Engle-Granger 协整检验")
    print("=" * 80)

    y = data[y_col].dropna()
    X = sm.add_constant(data[x_cols].dropna())
    common_idx = y.index.intersection(X.index)
    y, X = y.loc[common_idx], X.loc[common_idx]

    reg = OLS(y, X).fit()
    resid = reg.resid
    adf_stat = adfuller(resid, regression="c", autolag="AIC")[0]

    n_vars = len(x_cols)
    eg_crit = [-3.04, -3.37, -3.70, -3.96][min(n_vars, 3)]
    is_coint = adf_stat < eg_crit

    print(f"被解释变量：{y_col}")
    print(f"解释变量：{', '.join(x_cols)}")
    print(f"残差ADF统计量：{adf_stat:.4f}")
    print(f"EG临界值(5%)：{eg_crit:.2f}")
    print("存在长期协整关系" if is_coint else "不存在长期协整关系")
    return is_coint, resid

def build_ecm_model(data, y_col, x_cols, resid):
    print("\n" + "=" * 80)
    print("误差修正模型 ECM")
    print("=" * 80)

    df_ecm = data[[y_col] + x_cols].copy().dropna()
    dy = df_ecm[y_col].diff()
    dX = df_ecm[x_cols].diff()
    ecm = resid.reindex(df_ecm.index).shift(1)

    df_model = pd.DataFrame({"dy": dy, "ecm": ecm}).join(dX).dropna()
    y_ecm, X_ecm = df_model["dy"], sm.add_constant(df_model.drop(columns=["dy"]))
    maxlags = int(len(y_ecm) ** 0.25)
    model = OLS(y_ecm, X_ecm).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    print(model.summary())
    return model

def fmols_robust(data, y_col, x_cols, kernel="bartlett", bandwidth=None):
    """
    Phillips–Hansen (1990) 完全修正最小二乘法 FMOLS
    用于 I(1) 协整变量的长期均衡稳健性估计，
    通过对扰动项与 Δx 的长期协方差进行非参数核估计，
    完成被解释变量的二阶矩修正与偏差修正项扣减。

    参数：
        data       : DataFrame，包含被解释变量与解释变量原序列
        y_col      : 被解释变量列名
        x_cols     : 解释变量列名列表（均应为 I(1)）
        kernel     : 'bartlett' (Newey–West) 或 'parzen'，默认 Bartlett
        bandwidth  : 带宽 M，默认按 floor(4*(T/100)^(2/9)) 自动选取
    """
    print("\n" + "=" * 80)
    print("稳健性检验：完全修正最小二乘法 FMOLS（Phillips–Hansen, 1990）")
    print("=" * 80)

    df_fm = data[[y_col] + x_cols].dropna().reset_index(drop=True)
    T = len(df_fm)
    k = len(x_cols)

    y = df_fm[y_col].values.astype(float)
    X_raw = df_fm[x_cols].values.astype(float)
    X_const = np.column_stack([np.ones(T), X_raw])

    # -------- 第 1 步：协整回归 OLS，得到初始残差 u_hat --------
    beta_ols = np.linalg.lstsq(X_const, y, rcond=None)[0]
    u_hat = y - X_const @ beta_ols

    # -------- 第 2 步：构造 ξ_t = (u_hat_t, Δx_t')，对齐到 T-1 个观测 --------
    dX = np.diff(X_raw, axis=0)            # (T-1) × k
    u_aligned = u_hat[1:]
    xi = np.column_stack([u_aligned, dX])  # (T-1) × (1+k)
    n = xi.shape[0]

    # -------- 第 3 步：核估计长期协方差矩阵 Ω 与单边协方差矩阵 Λ --------
    if bandwidth is None:
        bandwidth = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))

    def _kernel_w(j, M, kind):
        x = j / (M + 1.0)
        if kind == "bartlett":
            return max(0.0, 1.0 - x)
        if kind == "parzen":
            ax = abs(x)
            if ax <= 0.5:
                return 1 - 6 * ax ** 2 + 6 * ax ** 3
            if ax <= 1.0:
                return 2 * (1 - ax) ** 3
            return 0.0
        return 0.0

    Sigma = (xi.T @ xi) / n          # j = 0
    Lambda = Sigma.copy()            # 单边 (j ≥ 0)
    Omega  = Sigma.copy()            # 双边
    for j in range(1, bandwidth + 1):
        w = _kernel_w(j, bandwidth, kernel)
        Gamma_j = (xi[j:].T @ xi[:-j]) / n
        Lambda += w * Gamma_j
        Omega  += w * (Gamma_j + Gamma_j.T)

    # -------- 第 4 步：分块 --------
    Omega_uu = float(Omega[0, 0])
    Omega_uv = Omega[0, 1:].reshape(1, k)
    Omega_vu = Omega[1:, 0].reshape(k, 1)
    Omega_vv = Omega[1:, 1:]

    Lambda_uv = Lambda[0, 1:].reshape(1, k)
    Lambda_vv = Lambda[1:, 1:]

    Omega_vv_inv = np.linalg.pinv(Omega_vv)
    Omega_uu_v = float(Omega_uu - (Omega_uv @ Omega_vv_inv @ Omega_vu))  # 条件长期方差

    # -------- 第 5 步：被解释变量二阶矩修正 y_plus = y - Ω_uv Ω_vv^{-1} Δx --------
    correction_y = np.zeros(T)
    correction_y[1:] = (Omega_uv @ Omega_vv_inv @ dX.T).flatten()
    y_plus = y - correction_y

    # -------- 第 6 步：偏差修正项 Δ_uv = Λ_uv - Ω_uv Ω_vv^{-1} Λ_vv --------
    delta_plus = (Lambda_uv - Omega_uv @ Omega_vv_inv @ Lambda_vv).flatten()

    # -------- 第 7 步：FMOLS 系数 = (X'X)^{-1} ( X' y_plus - T·[0, Δ_uv]' ) --------
    bias_vec = np.concatenate([[0.0], delta_plus]) * T
    XtX_inv = np.linalg.inv(X_const.T @ X_const)
    beta_fm = XtX_inv @ (X_const.T @ y_plus - bias_vec)

    # -------- 第 8 步：FMOLS 标准误 Var(β) = Ω_{u·v} (X'X)^{-1} --------
    var_beta_fm = Omega_uu_v * XtX_inv
    se_fm = np.sqrt(np.maximum(np.diag(var_beta_fm), 0.0))
    z_fm  = beta_fm / np.where(se_fm == 0, np.nan, se_fm)
    p_fm  = 2 * (1 - norm.cdf(np.abs(z_fm)))

    # -------- 第 9 步：拟合优度（仍以 y 自身为基准） --------
    y_hat = X_const @ beta_fm
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (T - 1) / max(1, (T - k - 1)) if not np.isnan(r2) else np.nan

    # -------- 输出 --------
    var_names = ["常数项"] + list(x_cols)
    res_df = pd.DataFrame({
        "变量":     var_names,
        "系数":     np.round(beta_fm, 4),
        "标准误":   np.round(se_fm, 4),
        "z统计量":  np.round(z_fm, 4),
        "P值":      np.round(p_fm, 4),
        "显著性":   [get_sig_stars(p) for p in p_fm],
    })

    print(f"样本量 T = {T}, 解释变量个数 k = {k}")
    print(f"核函数：{kernel}    带宽 M = {bandwidth}")
    print(f"长期方差         Ω_uu   = {Omega_uu:.4f}")
    print(f"条件长期方差     Ω_(u·v) = {Omega_uu_v:.4f}")
    print(f"R² = {r2:.4f},  Adj R² = {adj_r2:.4f}")
    print("\nFMOLS 估计结果：")
    print(res_df.to_string(index=False))
    print("\n注：*** p<0.01, ** p<0.05, * p<0.1")

    # 与原代码兼容：返回与原函数同结构的简表
    return res_df

def get_sig_stars(pval):
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.1:
        return "*"
    return ""

def run_comprehensive_analysis(file_path):
    print("=" * 80)
    print("毕业论文实证分析")
    print("=" * 80)

    if not os.path.exists(file_path):
        print(f"文件不存在")
        return

    df = pd.read_excel(file_path)
    print("数据读取成功")

    first_col = df.columns[0]
    if "Unnamed" in str(first_col) or df[first_col].dtype == "object":
        temp_year = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        df["Year"] = temp_year.ffill().bfill()
        df = df.dropna(subset=["Year"])
    else:
        df["Year"] = df[first_col]
    df = df.sort_values("Year").reset_index(drop=True)

    required_cols = ["中国", "美国", "德国", "日本", "企业个数", "专利数", "R&D经费内部支出",
                     "全部从业人员平均人数", "主营业务收入", "资产合计", "外商资本金", "经济自由度因素"]

    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="linear")
            if not df[col].isna().all():
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = 0

    df["China_GVC"] = np.log(df["中国"].abs() + 1e-6)
    target_col = "China_GVC"

    tech_countries = ["美国", "德国", "日本"]
    if all(c in df.columns for c in tech_countries):
        X = df[tech_countries].dropna()
        X_scaled = StandardScaler().fit_transform(X)
        eig_vals, eig_vecs = np.linalg.eig(MinCovDet().fit(X_scaled).covariance_)
        idx = eig_vals.argsort()[::-1]
        df.loc[X.index, "Tech_Spillover"] = X_scaled @ eig_vecs[:, idx[0]]
        df["Tech_Spillover"] = df["Tech_Spillover"].interpolate().bfill().ffill()

    def safe_div(n, d):
        d = d.replace(0, np.nan).ffill().bfill().fillna(1e-8)
        return n / d

    emp, rd, revenue, asset, patent, firm_num = [df[c] for c in ["全部从业人员平均人数","R&D经费内部支出","主营业务收入","资产合计","专利数","企业个数"]]
    df["RD_Intensity"] = np.log(safe_div(rd, revenue).abs() + 1e-6)
    df["Labor_Scale"] = safe_div(emp, firm_num)
    df["EcoFree"] = np.log(df["经济自由度因素"].abs() + 1e-6)
    df["Innov_Eff"] = np.log(patent.abs() + 1e-6)
    df["Foreign_Dep"] = np.log(safe_div(df["外商资本金"], asset).abs() + 1e-6)
    df["Firm_N"] = np.log(firm_num.abs() + 1e-6)

    scale_cols = ["RD_Intensity", "Labor_Scale", "Foreign_Dep", "EcoFree", "Innov_Eff", "Firm_N", "Tech_Spillover"]
    for c in scale_cols:
        if c in df.columns:
            df[c] = StandardScaler().fit_transform(df[[c]])

    print("=" * 80)
    print("一阶差分生成")
    print("=" * 80)
    diff_vars = [target_col, "RD_Intensity", "Labor_Scale", "Tech_Spillover", "EcoFree", "Innov_Eff", "Foreign_Dep","Firm_N"]
    for v in diff_vars:
        if v in df.columns:
            df[f"{v}_diff1"] = df[v].diff()
            print(f"{v}_diff1")

    core_vars = ["RD_Intensity", "Labor_Scale", "Tech_Spillover", "Foreign_Dep"]
    minor_vars = ["EcoFree", "Innov_Eff"]
    model_data = df.dropna().reset_index(drop=True)
    print(f"\n样本：{int(model_data['Year'].min())}–{int(model_data['Year'].max())}，共{len(model_data)}年")

    print("\n" + "=" * 80)
    print("相关性矩阵")
    print("=" * 80)
    corr = model_data[core_vars].corr().round(2)
    print(corr)

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("相关性热力图")
    plt.tight_layout()
    plt.savefig("论文图表/相关性.png", dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print("描述性统计")
    print("=" * 80)
    desc = model_data[core_vars].describe().round(4)
    print(desc)

    print("\n" + "=" * 80)
    print("VIF 多重共线性")
    print("=" * 80)
    vif = calculate_vif(model_data[core_vars])
    print(vif.to_string(index=False))

    print("\n" + "=" * 80)
    print("原序列平稳性检验")
    print("=" * 80)
    adf_list = [target_col] + core_vars + minor_vars
    for v in adf_list:
        if v in model_data.columns:
            adf_test(model_data[v], v, is_diff=False)

    print("\n" + "=" * 80)
    print("一阶差分平稳性检验")
    print("=" * 80)
    for v in adf_list:
        vd = f"{v}_diff1"
        if vd in df.columns:
            adf_test(df[vd].dropna(), vd, is_diff=True)

    print("\n" + "=" * 80)
    print("单整性 I(1) 检验")
    print("=" * 80)
    valid_i1 = []
    for v in adf_list:
        vd = f"{v}_diff1"
        if v in model_data.columns and vd in df.columns:
            if check_i1(v, vd):
                valid_i1.append(v)
    print(f"\n可协整I(1)变量：{valid_i1}")

    print("\n" + "=" * 80)
    print("协整检验")
    print("=" * 80)
    eg_vars = [v for v in core_vars if v in valid_i1]
    is_coint, resid, model = False, None, None
    if len(eg_vars) >= 1 and target_col in valid_i1:
        is_coint, resid = eg_test(model_data, target_col, eg_vars)
    else:
        print("有效I(1)变量不足，跳过协整")

    print("\n" + "=" * 80)
    print("模型选择")
    print("=" * 80)
    diff_core = [f"{v}_diff1" for v in core_vars if f"{v}_diff1" in model_data.columns]

    if is_coint:
        print("存在协整 → ECM模型")
        model = build_ecm_model(model_data, target_col, core_vars, resid)
    else:
        print("无协整 → 一阶差分OLS")
        y = model_data[f"{target_col}_diff1"]
        X = sm.add_constant(model_data[diff_core])
        model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        print(model.summary())

    if model is not None:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
        ax1.scatter(model.fittedvalues, model.resid)
        ax1.axhline(0, c="r", ls="--")
        ax1.set_title("残差散点图")
        sm.qqplot(model.resid, line="45", fit=True, ax=ax2)
        ax2.set_title("正态QQ图")
        plt.tight_layout()
        plt.savefig("论文图表/残差图.png", dpi=300)
        plt.close()
        print("\n图表已保存")

    print("\n" + "=" * 80)
    print("自相关 & 异方差检验")
    print("=" * 80)
    if model is not None:
        lb = acorr_ljungbox(model.resid, lags=[1,2,3,4], return_df=True)
        print("Ljung-Box 自相关检验:\n", lb.round(3))
        bp = het_breuschpagan(model.resid, model.model.exog)
        print(f"BP异方差检验：统计量={bp[0]:.2f}, p={bp[1]:.3f}")

    fmols_df = fmols_robust(model_data, target_col, core_vars)

    with pd.ExcelWriter("实证结果全表.xlsx") as writer:
        desc.to_excel(writer, sheet_name="描述统计")
        corr.to_excel(writer, sheet_name="相关系数")
        vif.to_excel(writer, sheet_name="VIF")
        if model is not None:
            pd.DataFrame([model.params, model.pvalues]).T.to_excel(writer, sheet_name="基准回归")
        if fmols_df is not None:
            fmols_df.to_excel(writer, sheet_name="FMOLS稳健")

    print("\n全部运行完成！结果已保存至 实证结果全表.xlsx")

if __name__ == "__main__":
    run_comprehensive_analysis(file_path)