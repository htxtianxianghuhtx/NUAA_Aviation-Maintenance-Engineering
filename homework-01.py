import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, lognorm, fisk

# 读取bearingcage数据
data = pd.DataFrame({
    'hours': [50, 150, 230, 250, 334, 350, 423, 450, 550, 650, 750, 850, 950, 990, 1009, 1050, 1150, 1250, 1350, 1450, 1510, 1550, 1650, 1850, 2050],
    'event': ['Censored', 'Censored', 'Failed', 'Censored', 'Failed', 'Censored', 'Failed', 'Censored', 'Censored', 'Censored', 'Censored', 'Censored', 'Censored', 'Failed', 'Failed', 'Censored', 'Censored', 'Censored', 'Censored', 'Censored', 'Failed', 'Censored', 'Censored', 'Censored', 'Censored'],
    'count': [288, 148, 1, 124, 1, 111, 1, 106, 99, 110, 114, 119, 127, 1, 1, 123, 93, 47, 41, 27, 1, 11, 6, 1, 2]
})

# 分离失败和删失数据
failed_data = data[data['event'] == 'Failed']
censored_data = data[data['event'] == 'Censored']

# 定义Weibull分布的负对数似然函数
def negative_log_likelihood_weibull(params, t, censored):
    shape, scale = params
    likelihoods = stats.weibull_min.pdf(t, shape, scale=scale)
    censored_likelihoods = stats.weibull_min.sf(censored, shape, scale=scale)
    total_likelihood = np.concatenate([likelihoods, censored_likelihoods])
    return -np.sum(np.log(total_likelihood))

# 定义lognormal分布的负对数似然函数
def negative_log_likelihood_lognormal(params, t, censored):
    mean, sigma = params
    likelihoods = stats.lognorm.pdf(t, s=sigma, scale=np.exp(mean))
    censored_likelihoods = stats.lognorm.sf(censored, s=sigma, scale=np.exp(mean))
    total_likelihood = np.concatenate([likelihoods, censored_likelihoods])
    return -np.sum(np.log(total_likelihood))

# 定义loglogistic分布的负对数似然函数
def negative_log_likelihood_loglogistic(params, t, censored):
    alpha, beta = params
    likelihoods = (beta / alpha) * (t / alpha) ** (beta - 1) / (1 + (t / alpha) ** beta)**2
    censored_likelihoods = 1 / (1 + (censored / alpha) ** beta)
    total_likelihood = np.concatenate([likelihoods, censored_likelihoods])
    return -np.sum(np.log(total_likelihood))

# 失败和删失时间
t_failed = failed_data['hours'].values
t_censored = censored_data['hours'].values

# 使用最小化函数拟合Weibull分布
initial_params_weibull = [1.5, 500]  # 初始参数估计
result_weibull = optimize.minimize(negative_log_likelihood_weibull, initial_params_weibull, args=(t_failed, t_censored), method='L-BFGS-B')
shape_weibull, scale_weibull = result_weibull.x

print(f"Weibull分布的形状参数: {shape_weibull:.4f}, 尺度参数: {scale_weibull:.4f}")

# 使用最小化函数拟合lognormal分布
initial_params_lognormal = [7.0, 1.0]  # 初始参数估计
result_lognormal = optimize.minimize(negative_log_likelihood_lognormal, initial_params_lognormal, args=(t_failed, t_censored), method='L-BFGS-B')
mean_lognormal, sigma_lognormal = result_lognormal.x

print(f"Lognormal分布的均值: {mean_lognormal:.4f}, 标准差: {sigma_lognormal:.4f}")

# 使用最小化函数拟合loglogistic分布
initial_params_loglogistic = [500, 1.5]  # 初始参数估计
result_loglogistic = optimize.minimize(negative_log_likelihood_loglogistic, initial_params_loglogistic, args=(t_failed, t_censored), method='L-BFGS-B')
alpha_loglogistic, beta_loglogistic = result_loglogistic.x

print(f"Loglogistic分布的alpha参数: {alpha_loglogistic:.4f}, beta参数: {beta_loglogistic:.4f}")

# 计算Weibull分布的MTTF (Mean Time To Failure)
MTTF_weibull = scale_weibull * stats.gamma(1 + 1/shape_weibull).mean()
print(f"Weibull分布的MTTF: {MTTF_weibull:.2f} 小时")

# 计算lognormal分布的MTTF
MTTF_lognormal = np.exp(mean_lognormal + (sigma_lognormal ** 2) / 2)
print(f"Lognormal分布的MTTF: {MTTF_lognormal:.2f} 小时")

# 计算loglogistic分布的MTTF (仅在beta > 1时存在)
if beta_loglogistic > 1:
    MTTF_loglogistic = alpha_loglogistic * (np.pi / beta_loglogistic) / np.sin(np.pi / beta_loglogistic)
    print(f"Loglogistic分布的MTTF: {MTTF_loglogistic:.2f} 小时")
else:
    print("Loglogistic分布的MTTF不存在 (beta <= 1)")

# 绘制不同分布的失效率在同一张图中
time_range = np.linspace(0, 3000, 1000)
plt.figure(figsize=(12, 8))

# Weibull失效率
hazard_rate_weibull = (shape_weibull / scale_weibull) * (time_range / scale_weibull) ** (shape_weibull - 1)
plt.plot(time_range, hazard_rate_weibull, label='Weibull')

# lognormal失效率
pdf_lognormal = stats.lognorm.pdf(time_range, s=sigma_lognormal, scale=np.exp(mean_lognormal))
cdf_lognormal = stats.lognorm.cdf(time_range, s=sigma_lognormal, scale=np.exp(mean_lognormal))
hazard_rate_lognormal = pdf_lognormal / (1 - cdf_lognormal)
plt.plot(time_range, hazard_rate_lognormal, label='Lognormal')

# loglogistic失效率
hazard_rate_loglogistic = (beta_loglogistic / alpha_loglogistic) * (time_range / alpha_loglogistic) ** (beta_loglogistic - 1) / (1 + (time_range / alpha_loglogistic) ** beta_loglogistic)
plt.plot(time_range, hazard_rate_loglogistic, label='Loglogistic')

plt.xlabel('Time (h)')
plt.ylabel('Hazard Rate')
plt.title('Hazard Rate Curves for Different Distributions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制不同分布的累计失效率在同一张图中
plt.figure(figsize=(12, 8))

# Weibull累计失效率
cumulative_hazard_weibull = 1 - np.exp(-((time_range / scale_weibull) ** shape_weibull))
plt.plot(time_range, cumulative_hazard_weibull, label='Weibull', linestyle='--')

# lognormal累计失效率
cumulative_hazard_lognormal = cdf_lognormal
plt.plot(time_range, cumulative_hazard_lognormal, label='Lognormal', linestyle='--')

# loglogistic累计失效率
cumulative_hazard_loglogistic = 1 / (1 + (time_range / alpha_loglogistic) ** (-beta_loglogistic))
plt.plot(time_range, cumulative_hazard_loglogistic, label='Loglogistic', linestyle='--')

plt.xlabel('Time (hours)')
plt.ylabel('Cumulative Hazard Rate')
plt.title('Cumulative Hazard Rate Curves for Different Distributions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#95%置信区间

weibull_std = weibull_min.std(shape_weibull, scale_weibull)
lognormal_std = lognorm.std(mean_lognormal,sigma_lognormal)
loglogistic_std = fisk.std(alpha_loglogistic,beta_loglogistic)


weibull_mttf = MTTF_weibull
lognormal_mttf = MTTF_lognormal
loglogistic_mttf = MTTF_loglogistic

confidence_level = 0.95
alpha = 1 - confidence_level

weibull_ci = (weibull_mttf - 1.96 * (weibull_std / np.sqrt(len(t_failed))),
              weibull_mttf + 1.96 * (weibull_std / np.sqrt(len(t_failed))))

lognormal_ci = (lognormal_mttf - 1.96 * (lognormal_std / np.sqrt(len(t_failed))),
                lognormal_mttf + 1.96 * (lognormal_std / np.sqrt(len(t_failed))))

loglogistic_ci = (loglogistic_mttf - 1.96 * (loglogistic_std / np.sqrt(len(t_failed))),
                  loglogistic_mttf + 1.96 * (loglogistic_std / np.sqrt(len(t_failed))))

mttf_results = {
    'Distribution': ['Weibull', 'LogNormal', 'LogLogistic'],
    '95% Lower': [weibull_ci[0], lognormal_ci[0], loglogistic_ci[0]],
    '95% Upper': [weibull_ci[1], lognormal_ci[1], loglogistic_ci[1]]
}
print("输出时精度控制：{:.2f}".format(weibull_ci[0]))
mttf_results_df = pd.DataFrame(mttf_results)
print("MTTF and 95% Confidence Intervals for Each Distribution:\n")
print(mttf_results_df)