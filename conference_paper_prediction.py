import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import chardet
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import warnings
from datetime import datetime
from sklearn.exceptions import ConvergenceWarning
import matplotlib.font_manager as fm
from adjustText import adjust_text

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

start_time = time.time()

print("正在检测文件编码...")

current_dir = os.getcwd()
csv_file = os.path.join(current_dir, "conference_data", "all_conferences_papers.csv")

try:
    with open(csv_file, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        file_encoding = result['encoding']
        confidence = result['confidence']
        print(f"检测文件编码为: {file_encoding} (置信度: {confidence:.2%})")

        if confidence < 0.8:
            print("置信度低, 尝试常见编码...")
            for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                try:
                    with open(csv_file, encoding=encoding) as test_f:
                        test_f.read(100)
                    print(f"成功鉴定编码为 {encoding}")
                    file_encoding = encoding
                    break
                except:
                    continue
except Exception as e:
    print(f"编码检测错误: {e}")
    file_encoding = 'utf-8'
    print(f"使用默认编码: {file_encoding}")

# 读取CSV数据集
print("\n正在读取数据...")
file_size = os.path.getsize(csv_file) / (1024 * 1024)

try:
    df = pd.read_csv(csv_file, encoding=file_encoding, on_bad_lines='warn')
    print(f"成功读取数据: {len(df):,} 条数据, 文件大小: {file_size:.2f} MB")
    print("数据列名:", df.columns.tolist())

except Exception as e:
    print(f"文件读取出错: {e}")
    print("尝试使用 UTF-8 和 Latin1 编码...")
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print("成功使用 UTF-8 编码")
    except:
        try:
            df = pd.read_csv(csv_file, encoding='latin1')
            print("成功使用 Latin1 编码")
        except Exception as e2:
            print(f"读取错误: {e2}")
            exit()

print("\n正在进行数据预处理...")

required_columns = ['Year', 'Conference_Type']
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    print(f"错误: 数据文件中缺少必要的列: {', '.join(missing_cols)}")
    print("可用列:", df.columns.tolist())
    exit()

try:
    df['Year'] = df['Year'].astype(int)
except Exception as e:
    print(f"年份转换错误: {e}")
    print("尝试从字符串中提取年份...")
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float).fillna(0).astype(int)

target_conferences = ['AAAI', 'IJCAI', 'CVPR', 'ICCV', 'ICML']
df = df[df['Conference_Type'].isin(target_conferences)]

current_year = pd.Timestamp.now().year
df = df[(df['Year'] >= 2020) & (df['Year'] <= current_year)]

print(f"过滤数据数: {len(df):,} 条")

print("\n统计每年每个会议的论文数量...")
count_df = df.value_counts(['Conference_Type', 'Year']).reset_index(name='Count')

print("创建时间框架...")
years = list(range(2020, current_year + 1))
all_combinations = pd.DataFrame(
    [(conf, year) for conf in target_conferences for year in years],
    columns=['Conference_Type', 'Year']
)

merged_df = pd.merge(all_combinations, count_df,
                     on=['Conference_Type', 'Year'],
                     how='left').fillna(0)
merged_df['Count'] = merged_df['Count'].astype(int)

pivot_df = merged_df.pivot(index='Year', columns='Conference_Type', values='Count')

# 会议周期信息
conference_cycles = {
    'AAAI': 1,
    'IJCAI': 1,
    'CVPR': 1,
    'ICCV': 2,  # ICCV会议在奇数年举办
    'ICML': 1
}

current_date = datetime.now()

# 论文数量预测函数（考虑会议周期）
def predict_conference_papers(conference, years, counts):
    # 转换成数组
    X = np.array(years).reshape(-1, 1)
    y = np.array(counts)

    results = {}
    last_year = max(years)
    cycle = conference_cycles[conference]
    next_year = last_year + cycle

    # 线性回归预测
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    linear_pred = max(0, int(linear_model.predict([[next_year]])[0]))
    results['Linear'] = linear_pred

    # 指数增长模型（带边界约束和归一化）
    if len(years) < 3:  # 数据点不足时使用线性预测
        results['Exponential'] = linear_pred
        print(f"使用线性预测构建 {conference} 指数模型 (数据不足)")
    else:
        try:
            # 归一化年份（从0开始）
            base_year = min(years)
            years_norm = np.array(years) - base_year
            next_year_norm = next_year - base_year

            def exp_growth(x, a, b):
                return a * np.exp(b * x)

            # 设置参数边界约束
            popt, _ = curve_fit(
                exp_growth,
                years_norm,
                counts,
                p0=[counts[0], 0.1],
                bounds=([0, 0], [np.inf, 1]),  # a>0, 0<=b<=1
                maxfev=2000
            )
            exp_pred = max(0, int(exp_growth(next_year_norm, *popt)))
            results['Exponential'] = exp_pred
        except Exception as e:
            print(f"指数拟合对 {conference} 未成功: {str(e)}")
            results['Exponential'] = linear_pred  # 失败时回退到线性预测

    # ARIMA时间序列预测
    if len(counts) < 3:  # ARIMA需要至少3个数据点
        results['ARIMA'] = results['Exponential']
        print(f"使用指数预测构建 {conference} ARIMA 模型 (数据不足)")
    else:
        try:
            model = ARIMA(y, order=(1, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.get_forecast(steps=cycle)
            arima_pred = max(0, int(forecast.predicted_mean[-1]))
            results['ARIMA'] = arima_pred
        except Exception as e:
            print(f"ARIMA 对 {conference} 未成功: {str(e)}")
            results['ARIMA'] = results['Exponential']

    # 预测值的加权平均
    valid_predictions = []
    if results['Linear'] > 0:
        valid_predictions.append(results['Linear'])
    if results['Exponential'] > 0:
        valid_predictions.append(results['Exponential'])
    if results['ARIMA'] > 0:
        valid_predictions.append(results['ARIMA'])

    if valid_predictions:
        avg_pred = int(np.mean(valid_predictions))
    else:
        # 后备方案：使用线性预测
        avg_pred = linear_pred

    results['Average'] = avg_pred
    results['Next Year'] = next_year

    return results

print("\nPredicting next conference paper counts...")
predictions = {}
prediction_df = pd.DataFrame(columns=['Conference', 'Next Year', 'Linear',
                                      'Exponential', 'ARIMA', 'Average'])

for conference in target_conferences:
    conf_data = merged_df[merged_df['Conference_Type'] == conference]

    conf_data = conf_data[conf_data['Count'] > 0]

    if len(conf_data) < 2:
        print(f"警告: 用于 {conference} 预测的现有数据不足")
        continue

    years = conf_data['Year'].values
    counts = conf_data['Count'].values

    pred_results = predict_conference_papers(conference, years, counts)
    predictions[conference] = pred_results

    row = {
        'Conference': conference,
        'Next Year': pred_results.get('Next Year', None),
        'Linear': pred_results.get('Linear', None),
        'Exponential': pred_results.get('Exponential', None),
        'ARIMA': pred_results.get('ARIMA', None),
        'Average': pred_results.get('Average', None)
    }
    prediction_df = pd.concat([prediction_df, pd.DataFrame([row])], ignore_index=True)

print("\n通过预测进行可视化处理...")
plt.figure(figsize=(16, 10))
plt.style.use('seaborn-v0_8-whitegrid')
markers = ['o', 's', '^', 'D', 'v']
colors = ['#0077dd', '#ff7f0e', '#39c5bb', '#d62728', '#8888cc']
texts = []  # 用于存储预测标签的文本对象

pred_marker = '*'
pred_size = 150

for i, conference in enumerate(target_conferences):
    conf_data = pivot_df[conference]
    valid_years = conf_data.index[conf_data > 0]

    if conference not in predictions:
        continue

    plt.plot(valid_years, conf_data.loc[valid_years],
             marker=markers[i], markersize=8, linewidth=2.5,
             label=conference, color=colors[i])

    # 获取最后一个有效数据点的年份和值
    last_year = valid_years[-1]  # 最后一个有数据的年份
    last_count = conf_data.loc[last_year]  # 该年份的论文数量

    # 获取预测年份和预测值
    next_year = predictions[conference]['Next Year']
    pred_value = predictions[conference]['Average']

    # 绘制虚线连接最后一年和预测点
    plt.plot(
        [last_year, next_year],
        [last_count, pred_value],
        linestyle='--',
        color=colors[i],
        linewidth=1.5
    )

    if predictions[conference]['Average'] is not None:
        pred_value = predictions[conference]['Average']
        next_year = predictions[conference]['Next Year']
        plt.scatter([next_year], [pred_value], marker=pred_marker, s=pred_size,
                    color=colors[i], edgecolors='k', zorder=5)

        text_obj = plt.text(
            next_year,
            pred_value + max(pivot_df.max().max() * 0.02, 5),
            f"{pred_value:.0f}",
            ha='center', va='bottom',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
        )
        texts.append(text_obj)  # 将文本对象添加到列表

    # 添加历史数据标签
    for year in valid_years:
        count = conf_data.loc[year]
        if count > 0:
            y_offset = max(pivot_df.max().max() * 0.02, 5)
            plt.text(year, count + y_offset, f"{count:,}",
                     ha='center', va='bottom', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

plt.title(f'Top Conference Publication Trends & Next Conference Predictions (2020-{current_year})',
          fontsize=16, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)

all_years = list(pivot_df.index)
for conference in predictions.values():
    if 'Next Year' in conference:
        all_years.append(conference['Next Year'])
all_years = sorted(set(all_years))
plt.xticks(all_years, fontsize=10)
plt.yticks(fontsize=10)

max_count = pivot_df.max().max()
max_pred = max([p['Average'] for p in predictions.values() if 'Average' in p] or [0])
plt.ylim(0, max(max_count, max_pred) * 1.25)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Conference', title_fontsize=12, fontsize=10,
           loc='upper left', frameon=True, shadow=True)
plt.figtext(0.5, 0.01,
            f"Data Source: DBLP | Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | Next Conference Predictions",
            ha="center", fontsize=9, color='gray')

# 调用 adjustText 自动调整标签位置
adjust_text(
    texts,  # 文本对象列表
    x=[p["Next Year"] for p in predictions.values()],
    y=[p["Average"] for p in predictions.values()],
    expand_text=(1.05, 1.05),  # 文本周围扩展范围
    expand_points=(1.05, 1.05),  # 数据点周围扩展范围
    force_text=(0.2, 0.5),  # 增加文本移动的力（水平方向0.05，垂直0.5）
    force_points=(0.2, 0.5),  # 增加数据点对标签的吸引力
    lim=50,  # 添加移动距离限制
    arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.5)
)

plt.tight_layout()
output_file = f'Conference_Paper_Trends_Next_Conference_Predictions_{current_year}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"带预测的图表已保存为 '{output_file}'")

print("\n按会议划分的年度论文数量:")
print(pivot_df.to_string())

print("\n下一届会议论文数量预测:")
print(prediction_df.to_string(index=False))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n成功完成处理! 已处理 {len(df):,} 条记录")
print(f"总时间: {elapsed_time:.2f} 秒")
print(f"处理速度: 每秒 {len(df) / elapsed_time:,.0f} 条记录")

plt.show()